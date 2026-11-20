# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Local candidate ranking for the optional RF-DETR picking-point head."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import Tensor

_MAX_SKELETON_POINTS = 32


def _centerline(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Thin a binary prediction with the two Zhang-Suen subiterations; return xy pixels."""
    y, x = np.nonzero(mask)
    if not len(x):
        return np.empty((0, 2), dtype=np.int64)
    x0, y0 = x.min(), y.min()
    grid = np.pad(mask[y0 : y.max() + 1, x0 : x.max() + 1], 1)
    center = grid[1:-1, 1:-1]
    while True:
        changed = False
        for phase in (0, 1):
            n, ne, e, se, s, sw, w, nw = ring = (
                grid[:-2, 1:-1],
                grid[:-2, 2:],
                grid[1:-1, 2:],
                grid[2:, 2:],
                grid[2:, 1:-1],
                grid[2:, :-2],
                grid[1:-1, :-2],
                grid[:-2, :-2],
            )
            count = np.sum(ring, axis=0, dtype=np.uint8)
            starts = np.sum([~a & b for a, b in zip(ring, ring[1:] + ring[:1])], axis=0, dtype=np.uint8)
            protected = ((n & e & s) | (e & s & w)) if phase == 0 else ((n & e & w) | (n & s & w))
            drop = center & (count >= 2) & (count <= 6) & (starts == 1) & ~protected
            if drop.any():
                center[drop] = False
                changed = True
        if not changed:
            break
    sy, sx = np.nonzero(center)
    if not len(sx):  # A tiny even-width island can disappear in a parallel thinning step.
        k = np.argmin((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
        return np.array([[x[k], y[k]]])
    return np.stack((sx + x0, sy + y0), axis=1)


def _locations(
    masks: NDArray[np.bool_], raw: NDArray[np.float32], ratio: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Keep E2 at slot zero and cover the predicted skeleton using farthest-point sampling."""
    count, h, w = masks.shape
    extent = np.array([w, h]) * ratio
    points = np.zeros((count, 1 + _MAX_SKELETON_POINTS, 2), dtype=np.float32)
    valid = np.zeros(points.shape[:2], dtype=bool)
    points[:, 0], valid[:, 0] = raw, True
    view_w, view_h = np.ceil(extent - 0.5).astype(int).clip(min=0)
    for row, mask in enumerate(masks):
        pixels = _centerline(mask[:view_h, :view_w])
        if not len(pixels):
            continue
        centers = pixels + 0.5
        squared = np.sum((centers - raw[row] * extent) ** 2, axis=1)
        # Start at the closest skeleton site for a reproducible local projection.
        k = int(squared.argmin())
        if squared[k] < 1e-10:
            if squared.max() < 1e-10:
                continue
            k = int(squared.argmax())
        for slot in range(1, points.shape[1]):
            points[row, slot] = centers[k] / extent
            valid[row, slot] = True
            squared = np.minimum(squared, np.sum((centers - centers[k]) ** 2, axis=1))
            k = int(squared.argmax())
            if squared[k] <= (extent.max() * 4 / 512) ** 2:
                break
    return points, valid


def rank_candidates(outputs: dict[str, Any], batch: int, queries: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return normalized candidate xy, valid slots and local scores for selected instance queries.

    Skeleton construction and seed coordinates are discrete proposals. Only query-conditioned local feature scores
    receive the ranking gradient. The existing E2 coordinate head retains its independent coordinate supervision.
    """
    spatial = outputs["candidate_spatial"][batch : batch + 1]
    query = outputs["candidate_query"][batch, queries].float()
    raw = outputs["pred_pick_points"][batch, queries].detach().float()
    ratio = outputs["candidate_valid_ratio"][batch].float()
    with torch.no_grad():
        masks = outputs["pred_masks"]
        if isinstance(masks, dict):
            logits = torch.einsum(
                "chw,nc->nhw", masks["spatial_features"][batch], masks["query_features"][batch, queries]
            )
            logits = logits + masks["bias"]
        else:
            logits = masks[batch, queries]
        # Interpolate logits before thresholding. Native mask cells can span several image pixels,
        # enough to truncate a thin stem during thinning. Bound temporary dense-mask memory by query.
        coords_list, valid_list = [], []
        raw_cpu, ratio_cpu = raw.cpu().numpy(), ratio.cpu().numpy()
        for start in range(0, len(queries), 8):
            chunk = logits[start : start + 8, None].float()
            fine = F.interpolate(chunk, scale_factor=4, mode="bilinear", align_corners=False)[:, 0] > 0
            coords, valid = _locations(fine.cpu().numpy(), raw_cpu[start : start + 8], ratio_cpu)
            coords_list.append(coords)
            valid_list.append(valid)
        if not coords_list:
            empty = raw.new_empty((0, 1 + _MAX_SKELETON_POINTS))
            return raw.new_empty((*empty.shape, 2)), empty.bool(), empty
        coords, valid = np.concatenate(coords_list), np.concatenate(valid_list)
    xy = torch.as_tensor(coords, device=query.device)
    available = torch.as_tensor(valid, device=query.device)
    grid = (xy * ratio * 2 - 1).unsqueeze(0)
    features = F.grid_sample(spatial.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    scores = (features[0].permute(1, 2, 0) * query[:, None]).sum(-1) / math.sqrt(query.shape[-1])
    return xy, available, scores.masked_fill(~available, -torch.inf)


def candidate_loss(xy: Tensor, valid: Tensor, scores: Tensor, truth: Tensor, size_hw: Tensor) -> Tensor:
    """Sum distance-supervised cross entropy; 5 pixels of softness at a 512-pixel long edge."""
    scale = size_hw.flip(0).float() / size_hw.max() * 512
    distance_sq = ((xy - truth[:, None]) * scale).square().sum(-1)
    target = (-distance_sq / (2 * 5**2)).masked_fill(~valid, -torch.inf).softmax(-1)
    log_prob = F.log_softmax(scores, dim=-1).masked_fill(~valid, 0)
    return -(target * log_prob).sum()
