# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU contracts for the per-instance topology-loss primitive."""

import pytest
import torch

from rfdetr.models.soft_cldice import soft_cldice_loss


class TestSoftClDiceLoss:
    """Check topology sensitivity, instance isolation, and numerical safety."""

    def test_accurate_thin_mask_has_small_loss(self) -> None:
        """An almost binary prediction of the target centerline has negligible loss."""
        target = torch.zeros(1, 11, 11)
        target[:, 1:10, 5] = 1
        logits = (target * 2 - 1) * 12
        assert soft_cldice_loss(logits, target).item() < 1e-3

    def test_gap_in_centerline_increases_loss(self) -> None:
        """Removing the middle of a stem is worse than preserving its connection."""
        target = torch.zeros(1, 11, 11)
        target[:, 1:10, 5] = 1
        correct = (target * 2 - 1) * 12
        broken = correct.clone()
        broken[:, 4:7, 5] = -12
        assert soft_cldice_loss(broken, target).item() > soft_cldice_loss(correct, target).item() + 0.05

    def test_instances_are_not_pooled_together(self) -> None:
        """Another instance's size or prediction cannot change an instance's loss."""
        target = torch.zeros(2, 11, 11)
        target[0, 1:10, 5] = 1
        target[1, 3:8, 3:8] = 1
        logits = (target * 2 - 1) * 6
        logits[1, 4:7, 4:7] = -6
        batched = soft_cldice_loss(logits, target)
        separate = torch.cat([soft_cldice_loss(logits[i : i + 1], target[i : i + 1]) for i in range(2)])
        torch.testing.assert_close(batched, separate, rtol=0, atol=0)

    def test_prediction_receives_finite_nonzero_gradient(self) -> None:
        """The soft skeleton and overlap terms remain differentiable."""
        target = torch.zeros(1, 9, 9)
        target[:, 1:8, 4] = 1
        logits = torch.zeros_like(target, requires_grad=True)
        soft_cldice_loss(logits, target).sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_targets_are_not_differentiated(self) -> None:
        """Ground-truth skeleton construction does not create a target gradient graph."""
        target = torch.ones(1, 5, 5, requires_grad=True)
        logits = torch.zeros_like(target, requires_grad=True)
        soft_cldice_loss(logits, target).sum().backward()
        assert target.grad is None

    def test_empty_matched_set_preserves_gradient_connection(self) -> None:
        """Ranks with no selected instances can reduce an empty result to a connected zero."""
        logits = torch.empty(0, 9, 9, requires_grad=True)
        result = soft_cldice_loss(logits, torch.empty_like(logits))
        assert result.shape == (0,)
        result.sum().backward()
        assert logits.grad is not None and logits.grad.shape == logits.shape

    @pytest.mark.parametrize("foreground", [0.0, 1.0])
    def test_uniform_targets_stay_finite(self, foreground: float) -> None:
        """Empty or fully filled targets do not cause a zero denominator."""
        target = torch.full((2, 7, 7), foreground)
        result = soft_cldice_loss(torch.full_like(target, -12), target)
        assert torch.isfinite(result).all()
        assert ((result >= 0) & (result <= 1)).all()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_precision_and_backward(self, dtype: torch.dtype) -> None:
        """Low-precision inputs compute reductions in FP32; FP64 remains available for checks."""
        logits = torch.zeros(1, 9, 9, dtype=dtype, requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, 1:8, 4] = 1
        result = soft_cldice_loss(logits, target)
        assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        result.sum().backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_cpu_autocast_keeps_reductions_in_fp32(self) -> None:
        """Mixed precision does not demote overlap sums to BF16."""
        target = torch.zeros(1, 9, 9)
        target[:, 1:8, 4] = 1
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = soft_cldice_loss(torch.zeros_like(target), target)
        assert result.dtype == torch.float32

    def test_double_precision_gradcheck(self) -> None:
        """Autograd agrees with finite differences away from pooling ties."""
        generator = torch.Generator().manual_seed(42)
        logits = torch.randn(1, 5, 5, generator=generator, dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, 1:4, 2] = 1
        assert torch.autograd.gradcheck(
            lambda value: soft_cldice_loss(value, target, iterations=2), (logits,), fast_mode=True
        )

    def test_inputs_are_unchanged(self) -> None:
        """Computing skeletons must not overwrite shared mask targets or logits."""
        logits = torch.zeros(1, 9, 9)
        target = torch.zeros_like(logits)
        target[:, 1:8, 4] = 1
        expected_logits, expected_target = logits.clone(), target.clone()
        soft_cldice_loss(logits, target)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(target, expected_target)

    def test_requires_instance_mask_shape(self) -> None:
        """The primitive accepts N,H,W masks, not a flattened set of random sample points."""
        with pytest.raises(ValueError, match="N, H, W"):
            soft_cldice_loss(torch.zeros(2, 10), torch.zeros(2, 10))

    def test_requires_matching_shapes(self) -> None:
        """Prediction and target grids must be aligned by the caller."""
        with pytest.raises(ValueError, match="same shape"):
            soft_cldice_loss(torch.zeros(1, 8, 8), torch.zeros(1, 9, 9))

    def test_rejects_negative_iterations(self) -> None:
        """A negative skeletonization budget is invalid."""
        with pytest.raises(ValueError, match="iterations"):
            soft_cldice_loss(torch.zeros(1, 9, 9), torch.zeros(1, 9, 9), iterations=-1)

    @pytest.mark.parametrize("smooth", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_smoothing(self, smooth: float) -> None:
        """The denominator stabilizer must be positive and finite."""
        with pytest.raises(ValueError, match="smooth"):
            soft_cldice_loss(torch.zeros(1, 9, 9), torch.zeros(1, 9, 9), smooth=smooth)
