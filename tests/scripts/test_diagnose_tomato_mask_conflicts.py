# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Validate the diagnostic boundary, native sampling preservation and additive attribution."""

import torch

from scripts.diagnose_tomato_decoder_gradients import feature_gradient
from scripts.diagnose_tomato_mask_conflicts import (
    attribute_mask_gradients,
    capture_mask_points,
    dense_probe_losses,
    geometry_stats,
    point_regions,
)
from tests.models.test_soft_cldice_integration import _batch, _criterion


def test_capture_and_region_vjps_reconstruct_native_loss_without_rng_changes() -> None:
    """Capture real CE/Dice on an FP32 toy feature map; summing region derivatives preserves both original gradients."""
    outputs, targets = _batch()
    features = torch.randn(1, 1, 3, 99, requires_grad=True)
    outputs["pred_masks"] = features[-1].reshape(1, 3, 9, 11)
    criterion = _criterion(0)
    criterion.weight_dict = {"loss_mask_ce": 5.0, "loss_mask_dice": 5.0}
    indices = criterion.matcher(outputs, targets)
    count = torch.tensor(3.0)
    torch.manual_seed(420)
    expected = criterion.loss_masks(outputs, targets, indices, count)
    rng = torch.random.get_rng_state()
    torch.manual_seed(420)
    with capture_mask_points() as captured:
        actual = criterion.loss_masks(outputs, targets, indices, count)
    assert torch.equal(rng, torch.random.get_rng_state())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    total = sum(actual[k] * criterion.weight_dict[k] for k in actual)
    grads = {"mask_420": feature_gradient(total, features)}
    result = attribute_mask_gradients(
        actual,
        captured,
        criterion,
        features,
        targets,
        indices,
        [dict(target_index=1)],
        outputs["mask_padding"],
        420,
        grads,
    )
    assert result["ce_plus_dice_relative_l2"] < 1e-6
    assert all(r["selected_feature_sum_relative_l2"] < 1e-6 for r in result["checks"])
    for name in ("ce", "dice"):
        parts = [g for k, g in grads.items() if k.startswith(name + "_") and k != name + "_420"]
        torch.testing.assert_close(sum(parts)[0, 1], grads[name + "_420"][0, 1], atol=1e-7, rtol=1e-6)


def test_regions_and_dense_errors_separate_center_omission_from_exterior_fp() -> None:
    """A known omission, extra pixel and saturated center have distinct diagnostic meanings."""
    labels = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    skeleton = torch.tensor([[1, 1, 0, 0, 0, 0, 0]])
    padding = torch.tensor([[0, 0, 0, 0, 0, 0, 1]])
    logits = torch.tensor([[-1, 1, -1, 1, 1, -1, 1.0]])
    regions = point_regions(labels, skeleton, padding, logits)
    assert torch.equal(torch.stack(list(regions.values()))[:, 0], torch.eye(7, dtype=torch.bool))
    truth = torch.zeros(5, 5, dtype=torch.bool)
    truth[1:4, 2] = True
    probability = truth.float()
    probability[2, 2] = 0
    probability[2, 0] = 1
    stats = geometry_stats(probability, truth, truth, torch.ones_like(truth))
    assert stats["center_missing"] == stats["fn_pixels"] == stats["fp_pixels"] == 1
    assert stats["center_saturated_pixels"] == 2
    assert stats["gt_dense_logit_gradient_l1"] == 0


def test_dense_objectives_have_expected_pixel_directions_and_exclude_padding() -> None:
    """The diagnostic must separately measure centerline omissions, actual FP, and all exterior mass."""
    p = torch.tensor([[[0.2, 0.8, 0.7, 0.1, 0.9]]], requires_grad=True)
    gt = torch.tensor([[True, True, False, False, False]])
    valid = torch.tensor([[True, True, True, True, False]])
    losses = dense_probe_losses(p, gt, gt, valid)
    expected = {
        "dense_center_missing": [-1.0, 0.0, 0.0, 0.0, 0.0],
        "dense_false_positive": [0.0, 0.0, 1.0, 0.0, 0.0],
        "dense_exterior": [0.0, 0.0, 0.5, 0.5, 0.0],
    }
    for key, gradient in expected.items():
        actual = torch.autograd.grad(losses[key].sum(), p, retain_graph=True)[0]
        torch.testing.assert_close(actual, torch.tensor([[gradient]]))
    empty = dense_probe_losses(p, gt, gt, torch.zeros_like(valid))
    assert all(value.item() == 0 for value in empty.values())
    assert torch.autograd.grad(sum(value.sum() for value in empty.values()), p)[0].count_nonzero() == 0
