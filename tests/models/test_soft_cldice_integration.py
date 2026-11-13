# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Class-selective topology loss: routing, geometry, gradients and CPU DDP."""

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.models.soft_cldice import soft_cldice_loss


class _IdentityMatcher:
    """Match targets in order, repeating them for each query group."""

    def __call__(self, outputs: dict[str, Any], targets: list[dict[str, Tensor]], group_detr: int = 1) -> list:
        """Return all identity pairs; exercises selection independently of matching randomness."""
        return [
            (torch.arange(len(t["labels"]) * group_detr), torch.arange(len(t["labels"])).repeat(group_detr))
            for t in targets
        ]


def _criterion(coef: float = 1.0, groups: int = 1) -> SetCriterion:
    """Build a mask criterion with stem class 1 selected.

    Examples:
        >>> _criterion().mask_cldice_class_ids
        (1,)
    """
    return SetCriterion(
        3,
        _IdentityMatcher(),
        {"loss_mask_cldice": coef},
        0.25,
        ["masks"],
        group_detr=groups,
        mask_cldice_class_ids=[1],
        mask_cldice_iterations=2,
    )


def _batch(labels: tuple[int, ...] = (0, 1, 2), height: int = 9, width: int = 11) -> tuple[dict, list]:
    """Create dense logits and vertical-line targets on a non-square grid.

    Examples:
        >>> _batch()[0]['pred_masks'].shape
        torch.Size([1, 3, 9, 11])
    """
    masks = torch.zeros(len(labels), height, width, dtype=torch.bool)
    masks[:, 1:-1, width // 2] = True
    outputs = {
        "pred_masks": torch.zeros(1, len(labels), height, width, requires_grad=True),
        "pred_logits": torch.zeros(1, len(labels), 3),
        "mask_padding": torch.zeros(1, height, width, dtype=torch.bool),
    }
    targets = [{"labels": torch.tensor(labels, dtype=torch.long), "masks": masks}]
    return outputs, targets


class TestClDiceConfig:
    """Opt-in settings reach the criterion without changing existing weights."""

    def test_defaults_off(self) -> None:
        """Existing segmentation configs do not register the new loss."""
        tc = SegmentationTrainConfig(dataset_dir="/tmp")
        assert tc.mask_cldice_loss_coef == 0
        assert tc.mask_cldice_class_ids == []
        assert tc.mask_cldice_chunk_size == 0
        criterion, _ = build_criterion_from_config(RFDETRSegNanoConfig(device="cpu"), tc)
        assert not any("cldice" in key for key in criterion.weight_dict)

    def test_config_roundtrip_and_final_layer_only(self) -> None:
        """Only one final-layer weight is registered; matcher costs stay unchanged."""
        tc = SegmentationTrainConfig(
            dataset_dir="/tmp",
            mask_cldice_loss_coef=0.2,
            mask_cldice_class_ids=[1],
            mask_cldice_iterations=3,
            mask_cldice_chunk_size=4,
        )
        restored = SegmentationTrainConfig.model_validate_json(tc.model_dump_json())
        mc = RFDETRSegNanoConfig(num_classes=3, device="cpu")
        ns = _namespace_from_configs(mc, restored)
        assert (ns.mask_cldice_loss_coef, ns.mask_cldice_class_ids, ns.mask_cldice_iterations) == (0.2, [1], 3)
        criterion, _ = build_criterion_from_config(mc, restored)
        base, _ = build_criterion_from_config(mc, SegmentationTrainConfig(dataset_dir="/tmp"))
        assert criterion.weight_dict == {**base.weight_dict, "loss_mask_cldice": 0.2}
        assert criterion.mask_cldice_class_ids == (1,)
        assert criterion.mask_cldice_iterations == 3
        assert ns.mask_cldice_chunk_size == criterion.mask_cldice_chunk_size == 4
        assert criterion.matcher.cost_mask_dice == base.matcher.cost_mask_dice
        assert criterion.state_dict().keys() == base.state_dict().keys()

    @pytest.mark.parametrize("coef", [-1, float("inf"), float("nan")])
    def test_invalid_weight(self, coef: float) -> None:
        """Reject nonfinite or negative coefficients."""
        with pytest.raises(ValueError):
            SegmentationTrainConfig(dataset_dir="/tmp", mask_cldice_loss_coef=coef, mask_cldice_class_ids=[1])

    @pytest.mark.parametrize("classes", [[], [-1], [1, 1]])
    def test_invalid_classes(self, classes: list[int]) -> None:
        """Enabled loss requires an explicit unique nonnegative selection."""
        with pytest.raises(ValueError):
            SegmentationTrainConfig(dataset_dir="/tmp", mask_cldice_loss_coef=1, mask_cldice_class_ids=classes)

    def test_class_outside_model(self) -> None:
        """A class id must exist in the actual dataset model configuration."""
        tc = SegmentationTrainConfig(dataset_dir="/tmp", mask_cldice_loss_coef=1, mask_cldice_class_ids=[3])
        with pytest.raises(ValueError, match="class"):
            build_criterion_from_config(RFDETRSegNanoConfig(num_classes=3, device="cpu"), tc)

    def test_invalid_iterations(self) -> None:
        """Negative skeleton depth is rejected at configuration time."""
        with pytest.raises(ValueError):
            SegmentationTrainConfig(dataset_dir="/tmp", mask_cldice_iterations=-1)

    def test_invalid_chunk_size(self) -> None:
        """A negative query chunk cannot define a valid computation partition."""
        with pytest.raises(ValueError):
            SegmentationTrainConfig(dataset_dir="/tmp", mask_cldice_chunk_size=-1)

    def test_sum_group_mode_rejected(self) -> None:
        """The initial experiment supports group averaging explicitly."""
        with pytest.raises(ValueError, match="sum_group_losses"):
            SetCriterion(
                3,
                _IdentityMatcher(),
                {"loss_mask_cldice": 1.0},
                0.25,
                ["masks"],
                sum_group_losses=True,
                mask_cldice_class_ids=[1],
            )


class TestClDiceRouting:
    """Original losses and their random-number consumption must be unchanged."""

    def test_zero_weight_bypasses_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zero weight needs neither metadata nor a topology calculation."""
        original = SetCriterion(3, _IdentityMatcher(), {}, 0.25, ["masks"])
        disabled = _criterion(0)
        monkeypatch.setattr(disabled, "loss_mask_cldice", lambda *args: pytest.fail("disabled branch executed"))
        outputs, targets = _batch()
        del outputs["mask_padding"]
        torch.manual_seed(5)
        expected = original(outputs, targets)
        expected_rng = torch.random.get_rng_state()
        expected_grad = torch.autograd.grad(sum(expected.values()), outputs["pred_masks"])[0]
        torch.manual_seed(5)
        actual = disabled(outputs, targets)
        actual_grad = torch.autograd.grad(sum(actual.values()), outputs["pred_masks"])[0]
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.random.get_rng_state())

    def test_enabled_preserves_original_and_aux_losses(self) -> None:
        """Topology runs after all original losses and only for final predictions."""
        outputs, targets = _batch()
        layer = {key: value for key, value in outputs.items() if key != "mask_padding"}
        outputs["aux_outputs"] = [layer]
        outputs["enc_outputs"] = layer
        torch.manual_seed(6)
        expected = _criterion(0)(outputs, targets)
        expected_rng = torch.random.get_rng_state()
        expected_grad = torch.autograd.grad(sum(expected.values()), outputs["pred_masks"])[0]
        torch.manual_seed(6)
        actual = _criterion()(outputs, targets)
        assert set(actual) - set(expected) == {"loss_mask_cldice"}
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        actual_grad = torch.autograd.grad(sum(actual[key] for key in expected), outputs["pred_masks"])[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.random.get_rng_state())

    def test_only_gt_stem_gets_topology_gradient(self) -> None:
        """GT class selection ignores deliberately incorrect predicted classes."""
        outputs, targets = _batch()
        outputs["pred_logits"][0, 0, 1] = 100
        outputs["pred_logits"][0, 1, 0] = 100
        loss = _criterion()(outputs, targets)["loss_mask_cldice"]
        loss.backward()
        grad = outputs["pred_masks"].grad[0]
        assert grad[1].abs().sum() > 0
        assert grad[[0, 2]].count_nonzero() == 0

    def test_selection_follows_nonidentity_matches(self) -> None:
        """A stem matched to query zero must supervise query zero, regardless of query order."""
        outputs, targets = _batch()
        indices = [(torch.tensor([2, 0, 1]), torch.tensor([0, 1, 2]))]
        loss = _criterion().loss_mask_cldice(outputs, targets, indices)["loss_mask_cldice"]
        loss.backward()
        grad = outputs["pred_masks"].grad[0]
        assert grad[0].abs().sum() > 0
        assert grad[[1, 2]].count_nonzero() == 0

    @pytest.mark.parametrize("labels", [(), (0, 2)])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_no_stem_is_connected_zero(self, labels: tuple, lazy: bool) -> None:
        """All segmentation tensors get zero gradients on empty local selections."""
        outputs, targets = _batch(labels)
        tensors = [outputs["pred_masks"]]
        if lazy:
            tensors = [
                torch.ones(1, 2, 9, 11, requires_grad=True),
                torch.ones(1, len(labels), 2, requires_grad=True),
                torch.zeros(1, requires_grad=True),
            ]
            outputs["pred_masks"] = dict(zip(("spatial_features", "query_features", "bias"), tensors))
        loss = _criterion()(outputs, targets)["loss_mask_cldice"]
        assert loss.item() == 0
        loss.backward()
        for tensor in tensors:
            assert tensor.grad is not None and tensor.grad.count_nonzero() == 0

    def test_group_repetition_does_not_scale_loss(self) -> None:
        """Identical repeated matches have the same loss as a single query group."""
        outputs, targets = _batch((1,))
        expected = _criterion()(outputs, targets)["loss_mask_cldice"]
        outputs["pred_masks"] = outputs["pred_masks"].repeat(1, 3, 1, 1)
        outputs["pred_logits"] = outputs["pred_logits"].repeat(1, 3, 1)
        actual = _criterion(groups=3)(outputs, targets)["loss_mask_cldice"]
        torch.testing.assert_close(actual, expected)

    def test_empty_gt_mask_is_excluded(self) -> None:
        """Masks emptied by geometry contribute neither loss nor normalization count."""
        outputs, targets = _batch((1, 1))
        targets[0]["masks"][1] = False
        loss = _criterion()(outputs, targets)["loss_mask_cldice"]
        expected = soft_cldice_loss(outputs["pred_masks"][0, :1], targets[0]["masks"][:1], iterations=2).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        assert outputs["pred_masks"].grad[0, 1].count_nonzero() == 0

    def test_missing_canvas_fails_clearly(self) -> None:
        """Do not guess canvas extent from target masks when block padding may exist."""
        outputs, targets = _batch()
        del outputs["mask_padding"]
        with pytest.raises(ValueError, match="mask_padding"):
            _criterion()(outputs, targets)

    def test_box_normalizer_override_rejected(self) -> None:
        """A box-count override cannot normalize a separate stem-count loss."""
        outputs, targets = _batch()
        with pytest.raises(ValueError, match="num_boxes"):
            _criterion()(outputs, targets, num_boxes=1.0)


class TestClDiceGeometry:
    """Dense and lazy masks share the full input canvas and valid image support."""

    def test_upsample_then_crop_non_square_padded_image(self) -> None:
        """Upsample to the padded canvas, not directly to each smaller GT extent."""
        outputs, targets = _batch((1,), height=7, width=9)
        outputs["pred_masks"] = torch.randn(1, 1, 5, 6, generator=torch.Generator().manual_seed(7), requires_grad=True)
        outputs["mask_padding"] = torch.ones(1, 10, 12, dtype=torch.bool)
        outputs["mask_padding"][:, :7, :9] = False
        actual = _criterion()(outputs, targets)["loss_mask_cldice"]
        aligned = F.interpolate(outputs["pred_masks"], (10, 12), mode="bilinear", align_corners=False)[0, :, :7, :9]
        expected = soft_cldice_loss(aligned, targets[0]["masks"], iterations=2).mean()
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual, outputs["pred_masks"])[0]
        reference_grad = torch.autograd.grad(expected, outputs["pred_masks"])[0]
        torch.testing.assert_close(grad, reference_grad)

    def test_padding_logits_have_no_loss_or_gradient(self) -> None:
        """Image padding inside a target canvas cannot become a false skeleton."""
        outputs, targets = _batch((1,))
        outputs["mask_padding"][:, :, :3] = True
        with torch.no_grad():
            outputs["pred_masks"][:, :, :, :3] = 100
        actual = _criterion()(outputs, targets)["loss_mask_cldice"]
        expected = soft_cldice_loss(
            outputs["pred_masks"][0].masked_fill(outputs["mask_padding"][0], -torch.inf),
            targets[0]["masks"],
            iterations=2,
        ).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        assert outputs["pred_masks"].grad[:, :, :, :3].count_nonzero() == 0

    def test_variable_image_extents_and_border_stem(self) -> None:
        """Each image uses its own GT extent, including instances touching the boundary."""
        outputs, targets = _batch((1,))
        outputs["pred_masks"] = torch.zeros(2, 1, 9, 11, requires_grad=True)
        outputs["pred_logits"] = torch.zeros(2, 1, 3)
        outputs["mask_padding"] = torch.zeros(2, 9, 11, dtype=torch.bool)
        outputs["mask_padding"][1, 5:] = True
        outputs["mask_padding"][1, :, 7:] = True
        smaller = torch.zeros(1, 5, 7, dtype=torch.bool)
        smaller[:, :, -1] = True
        targets.append({"labels": torch.tensor([1]), "masks": smaller})
        expected = (
            soft_cldice_loss(outputs["pred_masks"][0], targets[0]["masks"], iterations=2).sum()
            + soft_cldice_loss(outputs["pred_masks"][1, :, :5, :7], smaller, iterations=2).sum()
        ) / 2
        # Exercise topology geometry directly: the original BCE/Dice requires pre-padded target masks.
        indices = [(torch.tensor([0]), torch.tensor([0])) for _ in targets]
        actual = _criterion().loss_mask_cldice(outputs, targets, indices)["loss_mask_cldice"]
        torch.testing.assert_close(actual, expected)

    def test_dense_lazy_value_and_gradients_match(self) -> None:
        """Reconstructing only selected queries reproduces dense-mask gradients."""
        outputs, targets = _batch()
        generator = torch.Generator().manual_seed(42)
        spatial = torch.randn(1, 2, 9, 11, generator=generator, requires_grad=True)
        query = torch.randn(1, 3, 2, generator=generator, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        tensors = (spatial, query, bias)
        outputs["pred_masks"] = torch.einsum("bchw,bqc->bqhw", spatial, query) + bias
        expected = _criterion()(outputs, targets)["loss_mask_cldice"]
        expected_grad = torch.autograd.grad(expected, tensors)
        outputs["pred_masks"] = dict(zip(("spatial_features", "query_features", "bias"), tensors))
        actual = _criterion()(outputs, targets)["loss_mask_cldice"]
        actual_grad = torch.autograd.grad(actual, tensors)
        torch.testing.assert_close(actual, expected)
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right)

    def test_bfloat16_autocast(self) -> None:
        """The integrated topology calculation keeps FP32 loss and finite gradients."""
        outputs, targets = _batch()
        outputs["pred_masks"] = outputs["pred_masks"].detach().to(torch.bfloat16).requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = _criterion()(outputs, targets)["loss_mask_cldice"]
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(outputs["pred_masks"].grad).all()


class _DDPMasks(nn.Module):
    """Minimal parameterized mask output for real CPU distributed backward."""

    def __init__(self) -> None:
        """Create a common mask parameter on both ranks."""
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 1, 9, 11))

    def forward(self) -> Tensor:
        """Return a non-leaf output for DDP reducer tracking."""
        return self.logits * 1


def _ddp_worker(rank: int, rendezvous: str, output_dir: str, labels: tuple[int, int], chunk_size: int) -> None:
    """Compare rank gradients after DDP reduction, including a rank without stems.

    Examples:
        Requires two spawned distributed processes:
        >>> _ddp_worker(0, 'file:///tmp/init', '/tmp', (1, 0), 4)  # doctest: +SKIP
    """
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        model = DistributedDataParallel(_DDPMasks())
        outputs, targets = _batch((labels[rank],))
        outputs["pred_masks"] = model().expand(-1, 13, -1, -1)
        outputs["pred_logits"] = outputs["pred_logits"].expand(-1, 13, -1)
        criterion = _criterion(groups=13)
        criterion.mask_cldice_chunk_size = chunk_size
        loss = criterion(outputs, targets)["loss_mask_cldice"]
        loss.backward()
        torch.save({"loss": loss.detach(), "grad": model.module.logits.grad}, Path(output_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestClDiceDDP:
    """Distributed mean must equal the corresponding serial stem-instance mean."""

    @pytest.mark.ddp
    @pytest.mark.parametrize("chunk_size", [0, 4])
    @pytest.mark.parametrize("labels", [(1, 1), (1, 0), (0, 0)])
    def test_cpu_two_rank_matches_serial(self, tmp_path: Path, labels: tuple[int, int], chunk_size: int) -> None:
        """All ranks participate even when their local selected count is zero."""
        torch.multiprocessing.spawn(
            _ddp_worker, args=((tmp_path / "init").as_uri(), str(tmp_path), labels, chunk_size), nprocs=2, join=True
        )
        logits = torch.zeros(1, 9, 11, requires_grad=True)
        _, targets = _batch((1,))
        expected = soft_cldice_loss(logits, targets[0]["masks"], iterations=2).mean()
        if 1 not in labels:
            expected = logits.sum() * 0
        expected.backward()
        records = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
        torch.testing.assert_close(sum(row["loss"] for row in records) / 2, expected.detach())
        for row in records:
            torch.testing.assert_close(row["grad"][0], logits.grad)
