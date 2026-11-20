# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exercise the E2 candidate opt-in at the mask, criterion and inference boundaries."""

import pytest
import torch

from rfdetr.config import RFDETRSegNanoConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.heads.local_candidates import rank_candidates
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.postprocess import PostProcess
from rfdetr.models.weights import _load_model_state
from tests.models.test_lwdetr_segmentation_eval import _build_seg_lwdetr, _images


def _outputs():
    masks = torch.full((1, 3, 12, 16), -1.0)
    masks[0, 0, 5, 1:15] = 1
    masks[0, 1] = masks[0, 0]
    return {
        "pred_logits": torch.tensor([[[9.0, -3.0], [-3.0, 8.0], [-4.0, 7.0]]]),
        "pred_boxes": torch.full((1, 3, 4), 0.5),
        "pred_masks": masks,
        "pred_pick_points": torch.full((1, 3, 2), 0.25, requires_grad=True),
        "candidate_spatial": torch.arange(16.0).expand(1, 1, 12, 16).clone().requires_grad_(),
        "candidate_query": torch.ones(1, 3, 1, requires_grad=True),
        "candidate_valid_ratio": torch.tensor([[0.75, 1.0]]),
    }


def test_raw_slot_and_valid_centerline_and_empty_mask():
    outputs = _outputs()
    xy, valid, scores = rank_candidates(outputs, 0, torch.arange(3))
    torch.testing.assert_close(xy[:, 0], outputs["pred_pick_points"][0])
    skeleton = xy[1, 1:][valid[1, 1:]]
    assert ((skeleton[:, 1] * 12 - 5.5).abs() < 0.5).all()
    assert len(skeleton) >= 6 and skeleton[:, 0].max() < 1
    assert valid[2].sum() == 1 and scores[2].argmax() == 0
    result = PostProcess(num_select=3, query_point_class_ids=[1])(outputs, torch.tensor([[120, 120]]))[0]
    assert result["pick_point_valid"].tolist() == [False, True, True]
    torch.testing.assert_close(result["pick_points"][[0, 2]], torch.full((2, 2), 30.0))
    torch.testing.assert_close(result["pick_points"][1], skeleton[skeleton[:, 0].argmax()] * 120)


@pytest.mark.parametrize("visible", [True, False])
def test_rank_loss_only_selected_stems_and_connected_empty_batch(visible):
    outputs = _outputs()
    target = {
        "labels": torch.tensor([0, 1, 1]),
        "size": torch.tensor([120, 120]),
        "keypoints": torch.tensor([[[0.2, 0.3, 2]], [[0.9, 0.45, 2 * visible]], [[0.1, 0.2, 0]]]),
    }
    criterion = SetCriterion(
        3, None, {"loss_query_point": 1.0, "loss_query_point_candidate": 1.0}, 0.25, [], query_point_class_ids=[1]
    )
    losses = criterion.loss_query_point(outputs, [target], [(torch.arange(3), torch.arange(3))])
    losses["loss_query_point_candidate"].backward()
    grads = outputs["candidate_query"].grad.flatten()
    assert (grads[1].abs().item() > 0) == visible and grads[[0, 2]].count_nonzero() == 0
    assert outputs["candidate_spatial"].grad is not None
    assert outputs["pred_pick_points"].grad is None


def test_warm_start_and_opt_in_native_masks():
    assert not RFDETRSegNanoConfig(device="cpu").query_point_candidate_head
    with pytest.raises(ValueError, match="requires E2"):
        RFDETRSegNanoConfig(query_point_candidate_head=True, device="cpu")
    stub = _build_seg_lwdetr()

    def build(enabled):
        return LWDETR(
            stub.backbone,
            stub.transformer,
            stub.segmentation_head,
            5,
            3,
            two_stage=True,
            aux_loss=True,
            query_point_head=True,
            query_point_candidate_head=enabled,
        ).eval()

    base, ranked = build(False), build(True)
    missing = _load_model_state(ranked, base.state_dict(), backbone_lora=False).missing_keys
    assert set(missing) == {"candidate_query_proj.weight", "candidate_query_proj.bias"}
    with torch.no_grad():
        ref = base(_images())
        for extra_masks in (False, True):
            out = ranked(_images(), all_layer_masks=extra_masks)
            for key in ("pred_masks", "pred_boxes", "pred_logits", "pred_pick_points"):
                assert torch.equal(ref[key], out[key])
            assert out["candidate_spatial"].shape[-2:] == out["pred_masks"].shape[-2:]
    incomplete = dict(ranked.state_dict())
    incomplete.pop("candidate_query_proj.bias")
    with pytest.raises(ValueError, match="candidate_query_proj.bias"):
        _load_model_state(ranked, incomplete, backbone_lora=False)
