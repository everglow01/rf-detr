# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Analytical CE behavior, padding exclusion and alignment through the real adapter."""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models.soft_cldice import _soft_skeleton, centerline_ce_loss
from tests.models.test_soft_cldice_integration import _criterion


@pytest.mark.parametrize("include_exterior", [False, True])
def test_signed_gradient_and_detached_supports(include_exterior: bool) -> None:
    """At zero logits each separately averaged support has exact derivative +/-0.5/count."""
    logits = torch.zeros(1, 1, 7, dtype=torch.float64, requires_grad=True)
    gt = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, True, True, False, True, True]])
    loss = centerline_ce_loss(logits, gt, gt, valid_mask=valid, include_exterior=include_exterior, exterior_radius=1)
    gradient, gt_gradient = torch.autograd.grad(loss.sum(), (logits, gt), allow_unused=True)
    expected = torch.zeros_like(logits)
    expected[..., 3] = -0.5
    expected[..., 2] = 0.5 if include_exterior else 0
    torch.testing.assert_close(gradient, expected)
    torch.testing.assert_close(loss, torch.tensor([math.log(2) * (1 + include_exterior)], dtype=torch.float64))
    assert gt_gradient is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_confident_errors_keep_finite_correction(dtype: torch.dtype) -> None:
    """Confident false negative/positive logits get nonvanishing CE; padded infinity is harmless."""
    logits = torch.tensor([[[-1000.0, 1000.0, -torch.inf]]], dtype=dtype, requires_grad=True)
    gt = torch.tensor([[[1.0, 0.0, 0.0]]])
    loss = centerline_ce_loss(logits, gt, gt, valid_mask=torch.tensor([[True, True, False]]), include_exterior=True)
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    assert loss.dtype == torch.float32 and torch.isfinite(loss).all() and torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.tensor([[[-1.0, 1.0, 0.0]]], dtype=dtype))


@pytest.mark.parametrize("count", [0, 2])
def test_empty_supports_are_connected_zero(count: int) -> None:
    """Absent positive and negative supports cannot produce NaN or disconnect the model."""
    logits = torch.zeros(count, 3, 3, requires_grad=True)
    gt = torch.zeros_like(logits)
    loss = centerline_ce_loss(logits, gt, gt, include_exterior=True)
    loss.sum().backward()
    assert loss.shape == (count,) and loss.count_nonzero() == 0
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


def test_adapter_resizes_to_gt_and_excludes_padding() -> None:
    """An odd, non-square image agrees with an independently resized masked BCE reference."""
    logits = torch.linspace(-2, 2, 12).reshape(1, 3, 4).requires_grad_()
    gt = torch.zeros(1, 5, 7, dtype=torch.bool)
    gt[:, 1:5, 3] = True
    padding = torch.zeros(6, 8, dtype=torch.bool)
    padding[4:, :] = True
    gt &= ~padding[:5, :7]
    skeleton = _soft_skeleton(gt.float()[:, None], 25)[:, 0]
    criterion = _criterion()
    criterion.mask_cldice_mode = "centerline_local_ce"
    actual = criterion._cldice_on_image(logits, gt, padding, skeleton, torch.tensor([0]))
    aligned = F.interpolate(logits[:, None], size=(6, 8), mode="bilinear", align_corners=False)[:, 0, :5, :7]
    center = skeleton.bool()
    outside = F.max_pool2d(gt.float()[:, None], 7, stride=1, padding=3)[:, 0].bool() & ~gt & ~padding[:5, :7]
    expected = F.binary_cross_entropy_with_logits(aligned[center], torch.ones_like(aligned[center]))
    expected += F.binary_cross_entropy_with_logits(aligned[outside], torch.zeros_like(aligned[outside]))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])
