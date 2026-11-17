# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Read-only attribution of native sampled CE/Dice gradients and dense stem errors."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.models import criterion as criterion_module
from rfdetr.models.soft_cldice import _soft_skeleton

REGIONS = (
    "center_missing",
    "center_covered",
    "region_missing",
    "region_covered",
    "outside_fp",
    "outside_other",
    "padding",
)


@contextmanager
def capture_mask_points() -> Iterator[dict[str, Tensor]]:
    """Observe the real native point tensors without changing their values, graph, sampling or loss formula."""
    captured: dict[str, Tensor] = {}
    original_ce = criterion_module.sigmoid_ce_loss_jit
    original_sample = criterion_module._sample_target_masks_at_points

    def ce(logits: Tensor, labels: Tensor, count: Tensor) -> Tensor:
        captured.update(logits=logits, labels=labels)
        return cast(Tensor, original_ce(logits, labels, count))

    def sample(targets: list[dict[str, Tensor]], indices: list[tuple[Tensor, Tensor]], coords: Tensor) -> Tensor:
        captured["coords"] = coords
        return original_sample(targets, indices, coords)

    with (
        patch.object(criterion_module, "sigmoid_ce_loss_jit", ce),
        patch.object(criterion_module, "_sample_target_masks_at_points", sample),
    ):
        yield captured


def point_regions(labels: Tensor, skeleton: Tensor, padding: Tensor, logits: Tensor) -> dict[str, Tensor]:
    """Partition points exactly once; labels are native nearest-neighbor binary GT, predictions use p>=0.5."""
    valid, fg, center, predicted = ~padding.bool(), labels.bool(), skeleton.bool(), logits.detach() >= 0
    assert not bool((center & ~fg).any())
    masks = dict(
        center_missing=valid & center & ~predicted,
        center_covered=valid & center & predicted,
        region_missing=valid & fg & ~center & ~predicted,
        region_covered=valid & fg & ~center & predicted,
        outside_fp=valid & ~fg & predicted,
        outside_other=valid & ~fg & ~predicted,
        padding=~valid,
    )
    assert bool((torch.stack(list(masks.values())).sum(0) == 1).all())
    return masks


def selected_matches(
    indices: list[tuple[Tensor, Tensor]], cases: list[dict[str, Any]]
) -> tuple[Tensor, Tensor, Tensor]:
    """Map each selected GT's repeated queries to native flattened matched-point rows."""
    positions, rows, offset = [], [], 0
    for bi, ((sources, truths), case) in enumerate(zip(indices, cases, strict=True)):
        matched = torch.where(truths == case["target_index"])[0].tolist()
        for row in matched:
            positions.append((bi, int(sources[row])))
            rows.append(offset + row)
        offset += len(sources)
    bi, qi = torch.tensor(positions).T
    return bi, qi, torch.tensor(rows)


def attribute_mask_gradients(
    losses: dict[str, Tensor],
    captured: dict[str, Tensor],
    criterion: Any,
    features: Tensor,
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],
    cases: list[dict[str, Any]],
    padding: Tensor,
    seed: int,
    gradients: dict[str, Tensor],
) -> dict[str, Any]:
    """Partition native point-logit derivatives then pull each part back to the same decoder features.

    Dice is differentiated as its original full ratio before partitioning, never recomputed on subsets. Region masks are
    detached. BF16 feature VJPs need not add bitwise, so record reconstruction residuals and check their magnitude.
    """
    bi, qi, rows = selected_matches(indices, cases)
    logits = captured["logits"][rows]
    labels, coords = captured["labels"][rows], captured["coords"][rows]
    simple_indices, skeleton_targets, padding_targets, truth_targets = [], [], [], []
    for image, (target, case) in enumerate(zip(targets, cases, strict=True)):
        truth = target["masks"][case["target_index"] : case["target_index"] + 1]
        height, width = truth.shape[-2:]
        valid_truth = truth.bool() & ~padding[image, :height, :width]
        skeleton = _soft_skeleton(valid_truth.float().unsqueeze(1), 25)[:, 0] > 0
        source = qi[bi == image]
        simple_indices.append((source, torch.zeros_like(source)))
        skeleton_targets.append(dict(masks=skeleton))
        truth_targets.append(dict(masks=truth))
        padding_targets.append(dict(masks=padding[image : image + 1, :height, :width]))
    with torch.no_grad():
        sample = criterion_module._sample_target_masks_at_points
        torch.testing.assert_close(sample(truth_targets, simple_indices, coords), labels, rtol=0, atol=0)
        regions = point_regions(
            labels,
            sample(skeleton_targets, simple_indices, coords),
            sample(padding_targets, simple_indices, coords),
            logits,
        )
    stats = []
    checks = []
    for name in ("ce", "dice"):
        loss = losses[f"loss_mask_{name}"] * criterion.weight_dict[f"loss_mask_{name}"]
        point_gradient = torch.autograd.grad(loss, captured["logits"], retain_graph=True)[0][rows].detach()
        total = torch.autograd.grad(loss, features, retain_graph=True)[0][-1].float().cpu()
        gradients[f"{name}_{seed}"] = total
        reconstructed = torch.zeros_like(total[bi, qi])
        for region, selection in regions.items():
            part = point_gradient * selection
            derivative = torch.autograd.grad(logits, features, grad_outputs=part, retain_graph=True)[0]
            assert torch.isfinite(derivative).all() and not bool(derivative[:-1].any())
            result = derivative[-1].float().cpu()
            gradients[f"{name}_{region}_{seed}"] = result
            reconstructed += result[bi, qi]
        reference = total[bi, qi]
        relative = float((reconstructed - reference).double().norm() / reference.double().norm().clamp(min=1e-30))
        assert relative < 0.03, (name, relative)
        checks.append(dict(loss=name, selected_feature_sum_relative_l2=relative))
        for index, (b, q) in enumerate(zip(bi.tolist(), qi.tolist(), strict=True)):
            stats.append(
                dict(
                    batch_image=b,
                    query=q,
                    loss=name,
                    points={key: int(mask[index].sum()) for key, mask in regions.items()},
                    point_gradient_l1={
                        key: float((point_gradient[index] * mask[index]).abs().sum()) for key, mask in regions.items()
                    },
                )
            )
    combined = gradients[f"ce_{seed}"] + gradients[f"dice_{seed}"]
    reference = gradients[f"mask_{seed}"]
    relative = float(
        (combined[bi, qi] - reference[bi, qi]).double().norm() / reference[bi, qi].double().norm().clamp(min=1e-30)
    )
    assert relative < 0.03, relative
    return dict(mask_seed=seed, checks=checks, ce_plus_dice_relative_l2=relative, points=stats)


def geometry_stats(probability: Tensor, truth: Tensor, skeleton: Tensor, valid: Tensor) -> dict[str, Any]:
    """Describe centerline omission, instance-exterior errors and saturated GT-only gradients on the dense grid."""
    prediction = (probability >= 0.5) & valid
    foreground = truth.bool() & valid
    center = skeleton.bool() & valid
    fp, fn = prediction & ~foreground, foreground & ~prediction
    near = F.max_pool2d(foreground.float()[None, None], 7, stride=1, padding=3)[0, 0].bool()
    p = probability[center]
    return dict(
        gt_pixels=int(foreground.sum()),
        tp_pixels=int((foreground & prediction).sum()),
        fn_pixels=int(fn.sum()),
        fp_pixels=int(fp.sum()),
        fp_within_3px=int((fp & near).sum()),
        fp_farther_3px=int((fp & ~near).sum()),
        center_pixels=int(center.sum()),
        center_missing=int((center & ~prediction).sum()),
        center_probability_mean=float(p.mean()) if p.numel() else None,
        center_probability_min=float(p.min()) if p.numel() else None,
        center_saturated_pixels=int((p == 1).sum()),
        gt_dense_logit_gradient_l1=float((p * (1 - p)).sum() / (center.sum() + 1)),
        gt_coverage_loss=float(1 - (p.sum() + 1) / (center.sum() + 1)),
    )


@torch.no_grad()
def dense_geometry(
    outputs: dict[str, Any],
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],
    cases: list[dict[str, Any]],
    padding: Tensor,
    images: Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure every selected query; save a GT-bbox+32px crop for group zero without selecting by outcome."""
    rows, crops = [], {}
    masks = outputs["pred_masks"]
    for bi, (case, target, (sources, truths)) in enumerate(zip(cases, targets, indices, strict=True)):
        queries = sources[truths == case["target_index"]].to(masks["query_features"].device)
        gt = target["masks"][case["target_index"]].bool()
        height, width = gt.shape
        valid = ~padding[bi, :height, :width]
        gt = gt & valid
        skeleton = _soft_skeleton(gt.float()[None, None], 25)[0, 0] > 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = (
                torch.einsum("chw,nc->nhw", masks["spatial_features"][bi], masks["query_features"][bi, queries])
                + masks["bias"]
            )
        logits = F.interpolate(logits.float()[:, None], size=padding.shape[-2:], mode="bilinear", align_corners=False)[
            :, 0, :height, :width
        ]
        probability = logits.masked_fill(~valid, -torch.inf).sigmoid()
        for query, p in zip(queries.tolist(), probability, strict=True):
            rows.append(dict(**case, batch_image=bi, query=query, **geometry_stats(p, gt, skeleton, valid)))
        y, x = torch.where(gt)
        y0, y1 = max(int(y.min()) - 32, 0), min(int(y.max()) + 33, height)
        x0, x1 = max(int(x.min()) - 32, 0), min(int(x.max()) + 33, width)
        rgb = images[bi].float().cpu()
        assert tuple(rgb.shape[-2:]) == tuple(padding.shape[-2:])
        rgb = (
            rgb * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ).clamp(0, 1)
        first = int(queries.argmin())
        prefix = f"pair{case['pair']}_{case['role']}"
        crops[prefix + "_rgb"] = (rgb[:, y0:y1, x0:x1].permute(1, 2, 0) * 255).byte().numpy()
        crops[prefix + "_truth"] = gt[y0:y1, x0:x1].byte().cpu().numpy()
        crops[prefix + "_skeleton"] = skeleton[y0:y1, x0:x1].byte().cpu().numpy()
        crops[prefix + "_probability"] = probability[first, y0:y1, x0:x1].half().cpu().numpy()
        crops[prefix + "_bounds"] = torch.tensor([y0, y1, x0, x1, int(queries[first])]).numpy()
    return rows, crops
