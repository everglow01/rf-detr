# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression checks for final matched mask-IoU classification supervision."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config


@pytest.mark.parametrize("chunk_size", [1, 16])
@pytest.mark.parametrize("sparse", [False, True])
def test_quality_geometry_and_gradient(chunk_size: int, sparse: bool) -> None:
    """Only genuine, nonempty stem matches receive BCE gradients with detached, unpadded IoU."""
    config = SegmentationTrainConfig(
        dataset_dir="/tmp", mask_quality_loss_coef=0.1, mask_quality_class_ids=[1], mask_quality_chunk_size=chunk_size
    )
    criterion, _ = build_criterion_from_config(RFDETRSegNanoConfig(num_classes=3, device="cpu"), config)
    scores = torch.full((1, 6, 4), -2.0, requires_grad=True)
    feature = torch.tensor([[[[1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]]], requires_grad=True)
    queries = torch.tensor([[[1.0], [-1.0], [1.0], [1.0], [1.0], [1.0]]], requires_grad=True)
    masks = (
        {"spatial_features": feature, "query_features": queries, "bias": torch.tensor(0.0)}
        if sparse
        else (torch.einsum("bchw,bnc->bnhw", feature, queries))
    )
    # Predictions cover columns 0,1. Column 1 is padding. One of the two valid predicted pixels is outside GT.
    padding = torch.tensor([[[False, True, False], [False, True, False]]])
    truth = torch.zeros(5, 2, 3, dtype=torch.bool)
    truth[:, 0, 0] = True
    truth[3].zero_()
    target = {"labels": torch.tensor([1, 0, 1, 1, 1]), "masks": truth, "valid": torch.tensor([1, 1, 0, 1, 1]).bool()}
    # Two queries from different groups can match the same GT. The last query is unmatched.
    indices = [(torch.arange(5), torch.tensor([0, 0, 1, 2, 3]))]
    outputs = {"pred_logits": scores, "pred_masks": masks, "mask_padding": padding}
    loss = criterion.loss_mask_quality(outputs, [target], indices)["loss_mask_quality"]
    reference = F.binary_cross_entropy_with_logits(scores[0, :2, 1], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(loss, reference)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.autograd.grad(reference, scores)[0])
    assert feature.grad is None and queries.grad is None


def test_quality_off_and_final_layer_only() -> None:
    """Public settings register exactly one weight and leave every native weight intact."""
    model = RFDETRSegNanoConfig(num_classes=3, device="cpu")
    baseline, _ = build_criterion_from_config(model, SegmentationTrainConfig(dataset_dir="/tmp"))
    enabled, _ = build_criterion_from_config(
        model, SegmentationTrainConfig(dataset_dir="/tmp", mask_quality_class_ids=[1], mask_quality_loss_coef=0.1)
    )
    assert "loss_mask_quality" not in baseline.weight_dict
    assert enabled.weight_dict == {**baseline.weight_dict, "loss_mask_quality": 0.1}


def test_background_batch_has_zero_gradient() -> None:
    """Batches without stems still give a connected zero scalar for backward and distributed averaging."""
    model = RFDETRSegNanoConfig(num_classes=3, device="cpu")
    criterion, _ = build_criterion_from_config(
        model, SegmentationTrainConfig(dataset_dir="/tmp", mask_quality_class_ids=[1], mask_quality_loss_coef=0.1)
    )
    scores = torch.randn(1, 2, 4, requires_grad=True)
    targets = [{"labels": torch.tensor([0]), "masks": torch.ones(1, 3, 3, dtype=torch.bool)}]
    outputs = {"pred_logits": scores, "pred_masks": torch.ones(1, 2, 3, 3), "mask_padding": torch.zeros(1, 3, 3).bool()}
    loss = criterion.loss_mask_quality(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))])["loss_mask_quality"]
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(scores.grad) == 0
