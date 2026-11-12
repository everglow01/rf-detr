# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GPU regression for the CPU indices produced by the Hungarian matcher."""

import pytest
import torch

from rfdetr.models.criterion import SetCriterion


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("has_stem", [False, True])
def test_cpu_matching_indices_with_cuda_masks(lazy: bool, has_stem: bool) -> None:
    """CPU matches must reproduce CUDA-index losses and gradients, including empty stem selections."""
    torch.manual_seed(7)
    criterion = SetCriterion(
        3,
        None,
        {"loss_mask_cldice": 0.1},
        0.25,
        ["masks"],
        mask_cldice_class_ids=[1],
        mask_cldice_iterations=3,
    )
    if lazy:
        tensors = [
            torch.randn(1, 2, 8, 10, device="cuda", requires_grad=True),
            torch.randn(1, 3, 2, device="cuda", requires_grad=True),
            torch.zeros(1, device="cuda", requires_grad=True),
        ]
        masks = dict(zip(("spatial_features", "query_features", "bias"), tensors))
    else:
        tensors = [torch.randn(1, 3, 8, 10, device="cuda", requires_grad=True)]
        masks = tensors[0]
    gt = torch.zeros(3, 8, 10, dtype=torch.bool, device="cuda")
    gt[:, 1:7, 4:6] = True
    targets = [{"masks": gt, "labels": torch.tensor([0, 1 if has_stem else 2, 2], device="cuda")}]
    outputs = {"pred_masks": masks, "mask_padding": torch.zeros(1, 8, 10, dtype=torch.bool, device="cuda")}
    cpu_indices = [(torch.tensor([2, 0, 1]), torch.tensor([0, 1, 2]))]
    gpu_indices = [(source.cuda(), target.cuda()) for source, target in cpu_indices]
    expected = criterion.loss_mask_cldice(outputs, targets, gpu_indices)["loss_mask_cldice"]
    actual = criterion.loss_mask_cldice(outputs, targets, cpu_indices)["loss_mask_cldice"]
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, tensors, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, tensors)
    for result, reference in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference)
    query_grad = actual_grad[1] if lazy else actual_grad[0]
    assert query_grad[0, [1, 2]].count_nonzero() == 0
    assert bool(query_grad[0, 0].abs().sum() > 0) == has_stem
