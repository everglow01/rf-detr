# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Independent picking-point supervision and joint segmentation inference."""

import pytest
import torch

from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import LWDETR, build_criterion_from_config
from rfdetr.models.postprocess import PostProcess
from rfdetr.models.weights import _load_model_state
from tests.models.test_lwdetr_segmentation_eval import _build_seg_lwdetr, _images
from tests.models.test_soft_cldice_integration import _IdentityMatcher


def _criterion(coef: float = 1.0) -> SetCriterion:
    """Build the isolated point criterion.

    Examples:
        >>> _criterion().query_point_class_ids
        (1,)
    """
    return SetCriterion(3, None, {"loss_query_point": coef}, 0.25, [], query_point_class_ids=[1])


def _model(enabled: bool) -> LWDETR:
    """Build identical shared weights and fixed decoder features for both switches.

    Examples:
        >>> _model(False).query_point_embed is None
        True
    """
    stub = _build_seg_lwdetr()
    torch.manual_seed(42)
    return LWDETR(
        stub.backbone,
        stub.transformer,
        stub.segmentation_head,
        5,
        3,
        aux_loss=True,
        two_stage=True,
        query_point_head=enabled,
    )


class TestQueryPointLoss:
    """Only visible matched selected-class points contribute, with group averaging."""

    def test_selection_group_mean_and_gradient(self) -> None:
        pred = torch.full((1, 4, 2), 0.5, requires_grad=True)
        target = {
            "labels": torch.tensor([0, 1, 1]),
            "keypoints": torch.tensor(
                [
                    [[0.0, 0.0, 2.0]],
                    [[0.1, 0.3, 2.0]],
                    [[0.9, 0.9, 0.0]],
                ]
            ),
        }
        indices = [(torch.arange(4), torch.tensor([0, 1, 2, 1]))]
        loss = _criterion().loss_query_point({"pred_pick_points": pred}, [target], indices)["loss_query_point"]
        torch.testing.assert_close(loss, torch.tensor(0.3))
        loss.backward()
        torch.testing.assert_close(pred.grad[0, [0, 2]], torch.zeros(2, 2))
        torch.testing.assert_close(pred.grad[0, [1, 3]], torch.full((2, 2), 0.25))

    @pytest.mark.parametrize("empty", [True, False])
    def test_empty_or_invisible_connected_zero(self, empty: bool) -> None:
        pred = torch.rand(1, 2, 2, requires_grad=True)
        n = 0 if empty else 1
        targets = [{"labels": torch.ones(n, dtype=torch.long), "keypoints": torch.zeros(n, 1, 3)}]
        indices = [(torch.arange(n), torch.arange(n))]
        loss = _criterion().loss_query_point({"pred_pick_points": pred}, targets, indices)["loss_query_point"]
        assert loss.item() == 0
        loss.backward()
        assert pred.grad is not None and pred.grad.count_nonzero() == 0

    def test_missing_annotation_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="keypoints"):
            _criterion().loss_query_point(
                {"pred_pick_points": torch.zeros(1, 1, 2)},
                [{"labels": torch.tensor([1])}],
                [(torch.tensor([0]), torch.tensor([0]))],
            )

    def test_zero_coefficient_preserves_original_losses_and_gradients(self) -> None:
        pred = torch.full((1, 1, 2), 0.5, requires_grad=True)
        criterion = _criterion(0.0)
        criterion.matcher = _IdentityMatcher()
        target = {"labels": torch.tensor([1]), "keypoints": torch.tensor([[[0.1, 0.3, 2.0]]])}
        output = {"pred_pick_points": pred, "pred_logits": torch.zeros(1, 1, 3), "pred_boxes": torch.zeros(1, 1, 4)}
        losses = criterion(output, [target])
        assert set(losses) == {"loss_query_point"}
        (losses["loss_query_point"] * criterion.weight_dict["loss_query_point"]).backward()
        assert pred.grad is not None and pred.grad.count_nonzero() == 0


class TestQueryPointModel:
    """Opt-in head consumes final shared queries without changing the existing RNG stream."""

    def test_shared_state_rng_and_outputs_identical(self) -> None:
        baseline = _model(False).eval()
        rng = torch.get_rng_state()
        enabled = _model(True).eval()
        assert torch.equal(rng, torch.get_rng_state())
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, enabled.state_dict()[name], rtol=0, atol=0)
        a, b = baseline(_images()), enabled(_images())
        for key in ("pred_logits", "pred_boxes", "pred_masks"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert b["pred_pick_points"].shape == (2, 3, 2)
        assert "pred_pick_points" not in a
        assert all("pred_pick_points" not in x for x in [*b["aux_outputs"], b["enc_outputs"]])

    def test_export_explicitly_unsupported(self) -> None:
        with pytest.raises(NotImplementedError, match="query.point|picking.point"):
            _model(True).forward_export(_images())

    def test_export_rejection_leaves_model_usable(self) -> None:
        """An unsupported export must fail before mutating normal inference state."""
        model = _model(True).eval()
        forward = model.forward
        with pytest.raises(NotImplementedError, match="query.point"):
            model.export()
        assert not model._export and model.forward == forward
        assert "pred_pick_points" in model(_images())

    def test_final_only_weight_and_independent_switch(self) -> None:
        mc = RFDETRSegNanoConfig(query_point_head=True, query_point_class_ids=[1], num_classes=3, device="cpu")
        criterion, _ = build_criterion_from_config(mc, SegmentationTrainConfig(dataset_dir="/tmp"))
        assert criterion.weight_dict["loss_query_point"] == 1
        assert not any(k.startswith("loss_query_point_") for k in criterion.weight_dict)
        assert "keypoints" not in criterion.losses and "masks" in criterion.losses
        assert not mc.use_grouppose_keypoints

    def test_pretrain_allows_only_whole_new_head_missing(self) -> None:
        base = _model(False).state_dict()
        enabled = _model(True)
        incompatible = _load_model_state(enabled, base, backbone_lora=False)
        assert all(k.startswith("query_point_embed.") for k in incompatible.missing_keys)
        broken = dict(base)
        broken.pop("class_embed.weight")
        with pytest.raises(ValueError, match="class_embed.weight"):
            _load_model_state(enabled, broken, backbone_lora=False)
        partial = enabled.state_dict()
        partial.pop("query_point_embed.layers.0.weight")
        with pytest.raises(ValueError, match="query_point_embed"):
            _load_model_state(enabled, partial, backbone_lora=False)
        _load_model_state(enabled, enabled.state_dict(), backbone_lora=False)
        with pytest.raises(ValueError, match="unexpected"):
            _load_model_state(_model(False), enabled.state_dict(), backbone_lora=False)


class TestQueryPointPostprocess:
    """Points, masks, boxes and labels follow exactly the same top-k and threshold rows."""

    @pytest.mark.parametrize("threshold", [None, 0.7, 1.0])
    def test_joint_selection(self, threshold: float | None) -> None:
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 1.0, -2.0], [3.0, 4.0, -3.0]]]),
            "pred_boxes": torch.full((1, 2, 4), 0.5),
            "pred_masks": torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
            "pred_pick_points": torch.tensor([[[0.1, 0.2], [0.6, 0.8]]]),
        }
        post = PostProcess(num_select=4, query_point_class_ids=[1])
        result = post(outputs, torch.tensor([[10, 20]]), score_threshold=threshold)[0]
        scores, labels, query = post._select_topk(outputs["pred_logits"])
        keep = torch.ones(4, dtype=torch.bool) if threshold is None else scores[0] > threshold
        torch.testing.assert_close(
            result["pick_points"], outputs["pred_pick_points"][0, query[0, keep]] * torch.tensor([20, 10])
        )
        assert torch.equal(result["pick_point_valid"], labels[0, keep] == 1)
        assert torch.equal(result["masks"].flatten(1).all(1), query[0, keep] == 1)
