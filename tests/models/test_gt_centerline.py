# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GT-only centerline supervision: analytical gradient, class isolation and disabled parity."""

import pytest
import torch

from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.models import criterion as criterion_module
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.models.soft_cldice import _soft_skeleton, gt_centerline_loss, region_centerline_loss
from tests.models.test_soft_cldice_integration import _batch, _criterion


def test_coverage_has_only_centerline_gradient_and_detached_gt() -> None:
    """Hand-computed coverage and derivative exclude an off-centerline false positive and GT gradients."""
    skeleton = torch.zeros(1, 5, 5, dtype=torch.float64)
    skeleton[:, 1:4, 2] = 1
    skeleton.requires_grad_()
    logits = torch.zeros_like(skeleton, requires_grad=True)
    loss = gt_centerline_loss(logits, skeleton)
    expected_gradient = -skeleton.detach() / 16
    actual_gradient, gt_gradient = torch.autograd.grad(loss.sum(), (logits, skeleton), allow_unused=True)
    torch.testing.assert_close(loss, torch.tensor([0.375], dtype=torch.float64))
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert gt_gradient is None


@pytest.mark.parametrize("count", [0, 2])
def test_empty_skeleton_is_connected_zero(count: int) -> None:
    """Empty GT centerlines neither penalize nor disconnect predicted masks."""
    logits = torch.zeros(count, 5, 5, requires_grad=True)
    loss = gt_centerline_loss(logits, torch.zeros_like(logits))
    loss.sum().backward()
    assert loss.shape == (count,) and loss.count_nonzero() == 0
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


@pytest.mark.parametrize("mode", ["gt_centerline", "region_centerline"])
def test_mode_roundtrip_registers_only_distinct_final_loss(mode: str) -> None:
    """The explicit mode reaches the criterion while all native weights and model state stay unchanged."""
    defaults = SegmentationTrainConfig(dataset_dir="/tmp")
    assert defaults.mask_cldice_mode == "cldice" and defaults.mask_cldice_loss_coef == 0
    enabled = SegmentationTrainConfig(
        dataset_dir="/tmp", mask_cldice_mode=mode, mask_cldice_loss_coef=0.1, mask_cldice_class_ids=[1]
    )
    restored = SegmentationTrainConfig.model_validate_json(enabled.model_dump_json())
    model = RFDETRSegNanoConfig(num_classes=3, device="cpu")
    criterion, _ = build_criterion_from_config(model, restored)
    baseline, _ = build_criterion_from_config(model, defaults)
    assert criterion.mask_cldice_mode == mode
    assert criterion.weight_dict == {**baseline.weight_dict, f"loss_mask_{mode}": 0.1}
    assert criterion.state_dict().keys() == baseline.state_dict().keys()


@pytest.mark.parametrize("mode", ["gt_centerline", "region_centerline"])
def test_zero_coefficient_preserves_values_gradients_rng(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Choosing the mode alone cannot execute a topology branch or change baseline loss sampling."""
    original = _criterion(0)
    disabled = _criterion(0)
    disabled.mask_cldice_mode = mode
    monkeypatch.setattr(disabled, "loss_mask_cldice", lambda *args: pytest.fail("disabled loss executed"))
    outputs, targets = _batch()
    del outputs["mask_padding"]
    torch.manual_seed(5)
    expected = original(outputs, targets)
    rng = torch.random.get_rng_state()
    grad = torch.autograd.grad(sum(expected.values()), outputs["pred_masks"])[0]
    torch.manual_seed(5)
    actual = disabled(outputs, targets)
    actual_grad = torch.autograd.grad(sum(actual.values()), outputs["pred_masks"])[0]
    assert actual.keys() == expected.keys() and torch.equal(rng, torch.random.get_rng_state())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, grad, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["gt_centerline", "region_centerline"])
@pytest.mark.parametrize("lazy", [False, True])
def test_stem_only_group_chunk_gradient_and_gt_reuse(lazy: bool, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thirteen repeated matches equal a direct reference; other classes receive no added gradient."""
    outputs, targets = _batch()
    logits = outputs["pred_masks"].detach().repeat(1, 13, 1, 1).requires_grad_()
    if lazy:
        spatial = torch.eye(99).reshape(1, 99, 9, 11)
        outputs["pred_masks"] = dict(
            spatial_features=spatial, query_features=logits.flatten(2), bias=(0 * logits.sum()).reshape(1)
        )
    else:
        outputs["pred_masks"] = logits
    criterion = _criterion(groups=13)
    criterion.mask_cldice_mode = mode
    criterion.mask_cldice_iterations = 25
    criterion.mask_cldice_chunk_size = 4
    criterion.weight_dict = {f"loss_mask_{mode}": 0.1}
    calls = []

    def record_gt(mask, iterations):
        assert not torch.is_grad_enabled() and not mask.requires_grad
        calls.append(mask.shape[0])
        return _soft_skeleton(mask, iterations)

    monkeypatch.setattr(criterion_module, "_soft_skeleton", record_gt)
    actual = criterion(outputs, targets)[f"loss_mask_{mode}"]
    skeleton = _soft_skeleton(targets[0]["masks"][1:2].float().unsqueeze(1), 25)[:, 0]
    expected = (
        gt_centerline_loss(logits[0, 1::3], skeleton.repeat(13, 1, 1))
        if mode == "gt_centerline"
        else region_centerline_loss(
            logits[0, 1::3], targets[0]["masks"][1:2].repeat(13, 1, 1), skeleton.repeat(13, 1, 1)
        )
    ).mean()
    grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grad, expected_grad)
    assert calls == [1]
    assert grad[:, 0::3].count_nonzero() == grad[:, 2::3].count_nonzero() == 0


def test_region_precision_and_coverage_analytical_gradient() -> None:
    """Centerline and interior logits rise, exterior logits fall; GT gets no derivative."""
    logits = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    skeleton = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    loss = region_centerline_loss(logits, target, skeleton)
    # P=4/5, R=3/4; dL/dP=-450/961, dL/dR=-512/961.
    gradient, target_gradient, skeleton_gradient = torch.autograd.grad(
        loss.sum(), (logits, target, skeleton), allow_unused=True
    )
    torch.testing.assert_close(loss, torch.tensor([7 / 31], dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor([[[-73 / 961, -9 / 961, 36 / 961]]], dtype=torch.float64))
    assert target_gradient is None and skeleton_gradient is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_region_precision_survives_saturated_centerline(dtype: torch.dtype) -> None:
    """A fulfilled centerline objective must still penalize exterior mass without a prediction skeleton."""
    logits = torch.tensor([[[100.0, 0.0]]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[[1.0, 0.0]]])
    loss = region_centerline_loss(logits, target, target)
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    assert loss.dtype == torch.float32 and loss.item() > 0
    assert gradient[0, 0, 0] == 0 and gradient[0, 0, 1] > 0
    empty = logits[:0]
    assert region_centerline_loss(empty, target[:0], target[:0]).shape == (0,)
    assert (
        torch.autograd.grad(region_centerline_loss(empty, target[:0], target[:0]).sum(), logits)[0].count_nonzero() == 0
    )
