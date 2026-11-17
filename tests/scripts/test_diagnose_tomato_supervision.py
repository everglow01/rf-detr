# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Check diagnostic gradient signs and that observing matching preserves assignments and RNG."""

import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from scripts.diagnose_tomato_supervision import capture_match, supervision


def test_native_supervision_separates_positive_negative_and_bad_box() -> None:
    """A good positive must rise, a duplicate must fall, and zero box overlap clamps the target."""
    criterion = SetCriterion(3, HungarianMatcher(), {"loss_ce": 2.0}, 0.25, ["labels"], ia_bce_loss=True)
    target = dict(labels=torch.tensor([1, 0, 1]), boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]] * 3))
    outputs = dict(
        pred_logits=torch.logit(torch.full((1, 4, 3), 0.1)),
        pred_boxes=torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]
        ),
    )
    indices = [(torch.tensor([0, 2, 3]), torch.tensor([0, 1, 2]))]
    state = supervision(criterion, outputs, target, indices)
    assert state["assigned"].tolist() == [0, -1, 1, 2]
    assert state["gradient"][0, 1] < 0
    assert (state["gradient"][1:, 1] > 0).all()
    torch.testing.assert_close(state["soft_targets"][0, 1], torch.tensor(0.1**0.25))
    torch.testing.assert_close(state["soft_targets"][3, 1], torch.tensor(0.01))
    for q in (0, 3):
        torch.testing.assert_close(
            state["gradient"][q, 1], (state["scores"][q, 1] - state["soft_targets"][q, 1]) * 2 / 3
        )
    analytic = supervision(criterion, outputs, target, indices, gradient_probe=False)
    torch.testing.assert_close(analytic["gradient"], state["gradient"])
    assert outputs["pred_logits"].grad is None


def test_capture_keeps_native_matching_rng_and_group_boundaries() -> None:
    """Inspection must use the original point draw and preserve each group's independent assignment."""
    torch.manual_seed(123)
    matcher = HungarianMatcher(cost_mask_ce=1, cost_mask_dice=1, mask_point_sample_ratio=4)
    outputs = dict(pred_logits=torch.randn(1, 8, 3), pred_boxes=torch.rand(1, 8, 4), pred_masks=torch.randn(1, 8, 8, 8))
    target = dict(labels=torch.tensor([0, 1]), boxes=torch.rand(2, 4), masks=torch.rand(2, 8, 8) > 0.5)
    torch.manual_seed(42)
    expected = matcher(outputs, [target], group_detr=2)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    actual, total, classes, audit = capture_match(matcher, outputs, target, 2)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for left, right in zip(expected[0], actual[0], strict=True):
        assert torch.equal(left, right)
    assert total.shape == classes.shape == (8, 2)
    assert audit["points"] == 16
    assert len(audit["foreground_points"]) == 2
    assert all(0 <= count <= 16 for count in audit["foreground_points"])
    assert sorted(actual[0][0].div(4, rounding_mode="floor").tolist()) == [0, 0, 1, 1]
