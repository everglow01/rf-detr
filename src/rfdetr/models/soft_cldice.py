# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-instance 2D soft-clDice, to complement the existing BCE and Dice losses.

The soft-skeleton operations follow Shit et al., CVPR 2021:
https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
Matching, class selection, spatial alignment, and distributed normalization
belong to the criterion adapter, not to this mathematical primitive.

The adapted soft-skeleton implementation retains its original MIT notice:

Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def _soft_erode(mask: Tensor) -> Tensor:
    """Apply the reference cross-shaped soft erosion to N,1,H,W masks."""
    vertical = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_skeleton(mask: Tensor, iterations: int) -> Tensor:
    """Accumulate differentiable skeleton layers after successive erosions."""
    opened = F.max_pool2d(_soft_erode(mask), kernel_size=3, stride=1, padding=1)
    skeleton = F.relu(mask - opened)
    for _ in range(iterations):
        mask = _soft_erode(mask)
        opened = F.max_pool2d(_soft_erode(mask), kernel_size=3, stride=1, padding=1)
        layer = F.relu(mask - opened)
        skeleton = skeleton + F.relu(layer - skeleton * layer)
    return skeleton


def soft_cldice_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    iterations: int = 10,
    smooth: float = 1.0,
    target_skeleton: Tensor | None = None,
    recall_beta: float = 1.0,
) -> Tensor:
    """Return one soft-clDice loss per matched instance, with no batch reduction.

    Args:
        logits: Predicted dense mask logits with shape (N, H, W).
        targets: Binary target masks on the same grid and device as logits.
        iterations: Number of erosions after the initial skeleton layer.
        smooth: Positive finite stabilizer for skeleton-overlap ratios.
        target_skeleton: Optional precomputed GT skeleton, shaped like targets and built with the same iterations.
            Detached like targets; lets repeated matches reuse GT preprocessing outside activation checkpointing.
        recall_beta: Positive F-beta balance. One keeps the original arithmetic; values above one favor GT
            skeleton coverage over predicted-skeleton precision, not object detection recall directly.

    Returns:
        Losses with shape (N,). FP16/BF16/FP32 inputs compute in FP32, while
        FP64 inputs retain FP64. An empty batch remains connected to logits.

    Raises:
        ValueError: Shapes disagree, masks are not 2D instance batches, or a
            skeletonization parameter is invalid.

    Notes:
        This is a topology supplement, not a replacement for the original
        region losses. Targets are detached; gradients flow only to logits.
        The caller must exclude image padding and align the full mask grid
        before this function. Random mask sample points have no grid topology.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (N, H, W)")
    if targets.shape != logits.shape:
        raise ValueError("logits and targets must have the same shape")
    if target_skeleton is not None and target_skeleton.shape != logits.shape:
        raise ValueError("target_skeleton and logits must have the same shape")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be positive and finite")

    if not math.isfinite(recall_beta) or recall_beta <= 0:
        raise ValueError("recall_beta must be positive and finite")

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    values = logits.to(dtype=dtype)
    if values.shape[0] == 0:
        return values.flatten(1).sum(1)
    probabilities = values.sigmoid().unsqueeze(1)
    with torch.no_grad():
        labels = targets.detach().to(dtype=dtype).unsqueeze(1)
        target_skeleton = (
            _soft_skeleton(labels, iterations)
            if target_skeleton is None
            else target_skeleton.detach().to(dtype=dtype).unsqueeze(1)
        )
    predicted_skeleton = _soft_skeleton(probabilities, iterations)
    dimensions = (1, 2, 3)
    precision = ((predicted_skeleton * labels).sum(dimensions) + smooth) / (predicted_skeleton.sum(dimensions) + smooth)
    sensitivity = ((target_skeleton * probabilities).sum(dimensions) + smooth) / (
        target_skeleton.sum(dimensions) + smooth
    )
    if recall_beta == 1.0:
        return 1 - 2 * precision * sensitivity / (precision + sensitivity)
    # Normalize coefficients before squaring to avoid overflow for finite beta.
    if recall_beta > 1.0:
        precision_weight, sensitivity_weight = 1.0, (1.0 / recall_beta) ** 2
    else:
        precision_weight, sensitivity_weight = recall_beta**2, 1.0
    return 1 - (precision_weight + sensitivity_weight) * precision * sensitivity / (
        precision_weight * precision + sensitivity_weight * sensitivity
    )


def gt_centerline_loss(logits: Tensor, target_skeleton: Tensor, *, smooth: float = 1.0) -> Tensor:
    """Return one minus GT centerline coverage, without a differentiable prediction skeleton.

    Args:
        logits: Dense mask logits, shaped (N, H, W), aligned to the valid GT grid.
        target_skeleton: Precomputed GT skeleton on the same grid, detached inside this function.
        smooth: Positive finite stabilizer, matching clDice's sensitivity term.

    Returns:
        One loss per instance. Empty skeletons contribute zero loss and gradient. BF16/FP16 compute in FP32.

    Notes:
        This is clDice's one-sided sensitivity loss, not an equivalent clDice implementation. It has no off-centerline
        false-positive penalty and must supplement native region losses. K applies only to upstream GT skeletonization;
        there is no beta parameter or prediction skeleton. The gradient remains connected to the segmentation head.
    """
    if logits.ndim != 3 or target_skeleton.shape != logits.shape:
        raise ValueError("logits and target_skeleton must have the same (N, H, W) shape")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be positive and finite")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probabilities = logits.to(dtype=dtype).sigmoid()
    skeleton = target_skeleton.detach().to(dtype=dtype)
    return 1 - ((probabilities * skeleton).sum((1, 2)) + smooth) / (skeleton.sum((1, 2)) + smooth)


def region_centerline_loss(
    logits: Tensor,
    targets: Tensor,
    target_skeleton: Tensor,
    *,
    smooth: float = 1.0,
    precision_weight: float = 1.0,
    margin: int = 0,
) -> Tensor:
    """Combine region precision and GT centerline coverage with a weighted harmonic score.

    All inputs have shape (N, H, W) on the valid GT grid. Targets and the precomputed GT skeleton are detached.
    FP16/BF16 compute in FP32, while FP64 stays FP64. Unlike clDice, precision uses the entire predicted probability
    mass, without predicting a skeleton. This is a distinct loss, not an equivalent optimization. Empty batches stay
    connected; an empty skeleton still permits a region penalty. The adapter excludes empty GT instances and padding
    before calling this function. Precision weight lies in [0, 1]: one preserves F1, zero is exactly GT-only coverage,
    and smaller values reduce precision relative to coverage. A nonnegative margin in GT-grid pixels excludes the
    exterior Chebyshev boundary band from precision only; it never rewards that band as GT. Margin zero preserves the
    previous loss exactly. Native CE/Dice must continue to use the original GT.
    """
    if logits.ndim != 3 or targets.shape != logits.shape or target_skeleton.shape != logits.shape:
        raise ValueError("logits, targets and target_skeleton must have the same (N, H, W) shape")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be positive and finite")
    if not math.isfinite(precision_weight) or not 0 <= precision_weight <= 1:
        raise ValueError("precision_weight must be finite and in [0, 1]")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if precision_weight == 0:
        return gt_centerline_loss(logits, target_skeleton, smooth=smooth)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probabilities = logits.to(dtype=dtype).sigmoid()
    labels = targets.detach().to(dtype=dtype)
    skeleton = target_skeleton.detach().to(dtype=dtype)
    precision_mass = probabilities
    if margin > 0 and labels.shape[0] > 0:
        with torch.no_grad():
            dilated = F.max_pool2d(labels[:, None], 2 * margin + 1, stride=1, padding=margin)[:, 0]
            band = dilated.bool() & ~labels.bool()
        precision_mass = probabilities.masked_fill(band, 0)
    precision = ((probabilities * labels).sum((1, 2)) + smooth) / (precision_mass.sum((1, 2)) + smooth)
    coverage = ((probabilities * skeleton).sum((1, 2)) + smooth) / (skeleton.sum((1, 2)) + smooth)
    if precision_weight == 1:
        return 1 - 2 * precision * coverage / (precision + coverage)
    return 1 - (1 + precision_weight) * precision * coverage / (precision + precision_weight * coverage)


def centerline_ce_loss(
    logits: Tensor,
    targets: Tensor,
    target_skeleton: Tensor,
    *,
    valid_mask: Tensor | None = None,
    include_exterior: bool = False,
    exterior_radius: int = 3,
) -> Tensor:
    """Return per-instance GT-centerline CE with optional, separately averaged local negative CE.

    Args:
        logits: Dense mask logits shaped (N, H, W), aligned to the original augmented GT grid.
        targets: Binary instance masks matching logits. Other instances remain negative for this query.
        target_skeleton: Precomputed GT skeleton matching logits; no prediction skeleton is used.
        valid_mask: Optional valid image mask, broadcastable to logits. Padding gets no loss or gradient.
        include_exterior: Add negative CE on dilate(GT, radius) minus GT, with equal internal weight.
        exterior_radius: Positive Chebyshev dilation radius in GT-grid pixels, used only for exterior CE.

    Returns:
        One loss per instance. Empty supports contribute connected zero; FP16/BF16 compute in FP32.

    Notes:
        This is a clCE-inspired experiment, not full clCE or an equivalent clDice optimization. Native region losses
        remain necessary. GT-derived supports are detached, but all predicted logits keep their gradient path.
    """
    if logits.ndim != 3 or targets.shape != logits.shape or target_skeleton.shape != logits.shape:
        raise ValueError("logits, targets and target_skeleton must have the same (N, H, W) shape")
    if exterior_radius < 1:
        raise ValueError("exterior_radius must be positive")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    values = logits.to(dtype=dtype)
    if values.shape[0] == 0:
        return values.flatten(1).sum(1)
    with torch.no_grad():
        valid = (
            torch.ones_like(targets, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.detach().bool().expand_as(targets)
        )
        labels = targets.detach().bool() & valid
        center = target_skeleton.detach().bool() & labels
    # Keep even unsupported padded logits finite before softplus, avoiding inf*0.
    values = values.masked_fill(~valid, 0)
    positive = F.softplus(-values).masked_fill(~center, 0).sum((1, 2)) / center.sum((1, 2)).clamp(min=1)
    if not include_exterior:
        return positive
    with torch.no_grad():
        dilated = F.max_pool2d(labels.to(dtype)[:, None], 2 * exterior_radius + 1, stride=1, padding=exterior_radius)[
            :, 0
        ]
        exterior = dilated.bool() & ~labels & valid
    negative = F.softplus(values).masked_fill(~exterior, 0).sum((1, 2)) / exterior.sum((1, 2)).clamp(min=1)
    return positive + negative
