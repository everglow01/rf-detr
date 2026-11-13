# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Recall-weighted clDice: coverage gradients, configuration and chunk equivalence."""

import pytest
import torch

from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.models.soft_cldice import soft_cldice_loss
from tests.models.test_soft_cldice_integration import _batch, _criterion


def test_recall_weight_strengthens_missing_centerline_gradient() -> None:
    """Increasing beta must push an omitted centerline pixel upward more strongly."""
    target = torch.zeros(1, 11, 11)
    target[:, 1:10, 5] = 1
    logits = torch.full_like(target, -8)
    logits[target.bool()] = 2
    logits[:, 5, 5] = -2
    logits.requires_grad_()
    standard = soft_cldice_loss(logits, target, iterations=25, recall_beta=1).sum()
    weighted = soft_cldice_loss(logits, target, iterations=25, recall_beta=2).sum()
    first = torch.autograd.grad(standard, logits)[0]
    second = torch.autograd.grad(weighted, logits)[0]
    assert weighted > standard
    assert second[0, 5, 5] < first[0, 5, 5] < 0


def test_recall_weight_retains_false_skeleton_penalty() -> None:
    """Favoring coverage must not remove the gradient suppressing an off-target skeleton."""
    target = torch.zeros(1, 11, 11)
    target[:, 1:10, 5] = 1
    logits = torch.full_like(target, -8)
    logits[target.bool()] = 2
    logits[:, 5, 2] = 2
    logits.requires_grad_()
    soft_cldice_loss(logits, target, iterations=25, recall_beta=2).sum().backward()
    assert logits.grad is not None and logits.grad[0, 5, 2] > 0


def test_recall_weight_gradcheck() -> None:
    """The weighted harmonic overlap remains differentiable away from pooling ties."""
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(1, 5, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, 1:4, 2] = 1
    assert torch.autograd.gradcheck(
        lambda x: soft_cldice_loss(x, target, iterations=2, recall_beta=2), (logits,), fast_mode=True
    )


@pytest.mark.parametrize("beta", [0, -1, float("inf"), float("nan")])
def test_invalid_recall_weight(beta: float) -> None:
    """A nonpositive or nonfinite balance cannot define the weighted score."""
    with pytest.raises(ValueError, match="recall_beta"):
        soft_cldice_loss(torch.zeros(1, 5, 5), torch.zeros(1, 5, 5), recall_beta=beta)


def test_recall_config_roundtrip_and_disabled_default() -> None:
    """The new scalar reaches the criterion while default configurations retain the native baseline."""
    default = SegmentationTrainConfig(dataset_dir="/tmp")
    assert default.mask_cldice_loss_coef == 0 and default.mask_cldice_recall_beta == 1
    enabled = default.model_copy(
        update=dict(mask_cldice_loss_coef=0.1, mask_cldice_class_ids=[1], mask_cldice_recall_beta=2)
    )
    restored = SegmentationTrainConfig.model_validate_json(enabled.model_dump_json())
    criterion, _ = build_criterion_from_config(RFDETRSegNanoConfig(num_classes=3, device="cpu"), restored)
    assert criterion.mask_cldice_recall_beta == 2
    assert not list(criterion.parameters())


def test_recall_weight_chunk_and_class_isolation() -> None:
    """Chunked Group DETR repetitions preserve weighted loss/gradients and exclude non-stem masks."""
    outputs, targets = _batch((0, 1, 2))
    original = outputs["pred_masks"]
    logits = original.detach().repeat(1, 13, 1, 1).requires_grad_()
    outputs["pred_masks"] = logits
    criterion = _criterion(groups=13)
    criterion.mask_cldice_iterations = 25
    criterion.mask_cldice_chunk_size = 4
    criterion.mask_cldice_recall_beta = 2
    actual = criterion.loss_mask_cldice(outputs, targets, [(torch.arange(39), torch.arange(3).repeat(13))])[
        "loss_mask_cldice"
    ]
    selected = torch.arange(1, 39, 3)
    expected = soft_cldice_loss(
        logits[0, selected], targets[0]["masks"][1:2].repeat(13, 1, 1), iterations=25, recall_beta=2
    ).mean()
    first = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    second = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(first, second)
    assert torch.count_nonzero(first[:, 0::3]) == torch.count_nonzero(first[:, 2::3]) == 0
