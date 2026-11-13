# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Loss/gradient equivalence when Group DETR shares unique GT skeletons."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from pytest_mock import MockerFixture

from rfdetr.models import criterion as criterion_module
from rfdetr.models.soft_cldice import _soft_skeleton, soft_cldice_loss
from tests.models.test_soft_cldice_integration import _batch, _criterion


@pytest.mark.parametrize("chunk_size", [0, 4])
@pytest.mark.parametrize("iterations", [10, 25])
@pytest.mark.parametrize("lazy", [False, True])
def test_group_dedup_preserves_loss_and_gradient(
    iterations: int, lazy: bool, chunk_size: int, mocker: MockerFixture
) -> None:
    """Reordered, repeated matches retain values and gradients without rebuilding GT on backward."""
    torch.manual_seed(91)
    outputs, targets = _batch((1, 0, 1, 1, 1, 2), height=31, width=35)
    masks = targets[0]["masks"]
    masks.zero_()
    masks[0, 2:28, 3:8] = True
    masks[3, 8:24, 17:24] = True
    masks[4, 2:20, 4:9] = True
    targets[0]["valid"] = torch.tensor([True, True, True, True, False, True])
    outputs["mask_padding"] = torch.zeros(1, 36, 40, dtype=torch.bool)
    outputs["mask_padding"][:, 26:, :] = True
    outputs["mask_padding"][:, :, 32:] = True
    if lazy:
        tensors = [
            torch.randn(1, 3, 7, 9, requires_grad=True),
            torch.randn(1, 78, 3, requires_grad=True),
            torch.zeros(1, requires_grad=True),
        ]
        outputs["pred_masks"] = dict(zip(("spatial_features", "query_features", "bias"), tensors))
        logits = torch.einsum("chw,nc->nhw", tensors[0][0], tensors[1][0]) + tensors[2]
    else:
        tensors = [torch.randn(1, 78, 7, 9, requires_grad=True)]
        outputs["pred_masks"] = tensors[0]
        logits = tensors[0][0]
    target_indices = torch.arange(6).repeat(13)
    source_indices = torch.randperm(78)
    selected = (target_indices == 0) | (target_indices == 3)
    expected_masks = masks[target_indices[selected]] & ~outputs["mask_padding"][0, :31, :35]
    aligned = F.interpolate(
        logits[source_indices[selected], None].float(), (36, 40), mode="bilinear", align_corners=False
    )[:, 0, :31, :35]
    aligned = aligned.masked_fill(outputs["mask_padding"][0, :31, :35], -torch.inf)
    expected = soft_cldice_loss(aligned, expected_masks, iterations=iterations).mean()
    expected_grads = torch.autograd.grad(expected, tensors, retain_graph=True)
    criterion = _criterion(groups=13)
    criterion.mask_cldice_iterations = iterations
    criterion.mask_cldice_chunk_size = chunk_size
    image_loss = mocker.spy(criterion, "_cldice_on_image")
    spy = mocker.spy(criterion_module, "_soft_skeleton")
    actual = criterion.loss_mask_cldice(outputs, targets, [(source_indices, target_indices)])["loss_mask_cldice"]
    actual_grads = torch.autograd.grad(actual, tensors)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-7, rtol=1e-6)
    assert spy.call_count == 1
    assert spy.call_args.args[0].shape == (2, 1, 31, 35)
    assert not spy.call_args.args[0].requires_grad
    if chunk_size:
        assert max(call.args[0].shape[0] for call in image_loss.call_args_list) <= chunk_size


def test_precomputed_skeleton_preserves_fp64_and_detaches_targets() -> None:
    """Cached GT is supervision, including when a caller gives it requires_grad=True."""
    torch.manual_seed(11)
    logits = torch.randn(2, 15, 17, dtype=torch.float64, requires_grad=True)
    masks = torch.zeros_like(logits)
    masks[:, 2:13, 6:10] = 1
    skeleton = _soft_skeleton(masks[:, None], 10)[:, 0].requires_grad_()
    expected = soft_cldice_loss(logits, masks, iterations=10).sum()
    actual = soft_cldice_loss(logits, masks.requires_grad_(), iterations=10, target_skeleton=skeleton).sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    result, target_grad, skeleton_grad = torch.autograd.grad(actual, [logits, masks, skeleton], allow_unused=True)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert target_grad is None and skeleton_grad is None


def test_precomputed_skeleton_rejects_mismatched_shape() -> None:
    """A broadcastable but wrong skeleton must not silently supervise another grid."""
    with pytest.raises(ValueError, match="target_skeleton"):
        soft_cldice_loss(torch.zeros(2, 9, 9), torch.zeros(2, 9, 9), target_skeleton=torch.zeros(1, 9, 9))
