# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Audit frozen tomato E1 artifacts without optimizer steps or model changes.

Run against E1's frozen source via PYTHONPATH. Each stage requires a new output folder. Augmentation seeds reproduce
distributions, not the historical worker RNG stream. Gradient measurements are local batch probes, not reconstructed DDP
steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from PIL import Image
from torch import Tensor

import rfdetr
from rfdetr.config import AugmentationBackend, MultiScale, RFDETRSegSmallConfig, SegmentationTrainConfig
from rfdetr.datasets.coco import CocoDetection
from rfdetr.models.soft_cldice import _soft_erode, _soft_skeleton
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou
from rfdetr.utilities.tensors import make_collate_fn


def write_json(path: Path, value: Any) -> None:
    """Write a diagnostic artifact, preserving non-ASCII paths and rejecting NaN."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_sample(seed: int, index: int) -> None:
    """Give each augmented sample an independent reproducible RNG stream."""
    value = seed * 1_000_000 + index
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


@torch.no_grad()
def skeleton_trace(masks: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compare K=10 against the same erosion algorithm run until extinction.

    Returns K=10 skeletons, reference skeletons, last contributing iteration, and an extinction flag. All-foreground
    masks are fixed points under the library's pooling boundary convention and are explicitly marked unresolved.
    """
    values = masks.float().unsqueeze(1)
    short = _soft_skeleton(values, 10)
    reference = _soft_skeleton(values, 0)
    last = torch.zeros(len(masks), device=masks.device, dtype=torch.long)
    remaining = values
    for iteration in range(1, max(masks.shape[-2:]) + 1):
        eroded = _soft_erode(remaining)
        layer = _soft_skeleton(eroded, 0)
        added = F.relu(layer - reference * layer)
        changed = added.flatten(1).any(1)
        last[changed] = iteration
        reference = reference + added
        fixed = torch.equal(eroded, remaining)
        remaining = eroded
        if fixed or not remaining.any():
            break
    extinct = ~remaining.flatten(1).any(1)
    return short[:, 0], reference[:, 0], last, extinct


def gradient_relation(first: list[Tensor | None], second: list[Tensor | None]) -> dict[str, Any]:
    """Measure concatenated parameter gradients, treating unused parameters as zero."""
    norm1 = sum(float(g.float().square().sum()) for g in first if g is not None)
    norm2 = sum(float(g.float().square().sum()) for g in second if g is not None)
    dot = sum(float(a.float().mul(b.float()).sum()) for a, b in zip(first, second) if a is not None and b is not None)
    finite = all(bool(torch.isfinite(g).all()) for g in first + second if g is not None)
    return dict(
        first_norm=math.sqrt(norm1),
        second_norm=math.sqrt(norm2),
        ratio=math.sqrt(norm2 / norm1) if norm1 else None,
        cosine=dot / math.sqrt(norm1 * norm2) if norm1 and norm2 else None,
        sum_norm=math.sqrt(max(0.0, norm1 + norm2 + 2 * dot)),
        finite=finite,
    )


def greedy_matches(ious: Tensor, scores: Tensor, eligible: Tensor, threshold: float) -> dict[int, int]:
    """Replicate confidence-greedy IoU>=0.5 matching without reusing a GT."""
    chosen: dict[int, int] = {}
    for query in torch.argsort(scores, descending=True, stable=True).tolist():
        if not eligible[query] or scores[query] < threshold:
            continue
        available = [i for i in range(ious.shape[1]) if i not in chosen]
        if available:
            target = max(available, key=lambda i: float(ious[query, i]))
            if ious[query, target] >= 0.5:
                chosen[target] = query
    return chosen


def configurations(args: argparse.Namespace) -> tuple[RFDETRSegSmallConfig, SegmentationTrainConfig]:
    """Load the saved E1 recipe, overriding only diagnostic execution settings."""
    cfg = yaml.safe_load(args.config.read_text())["model"]
    mc = dict(cfg["model_config"]["init_args"])
    if args.checkpoint:
        mc["pretrain_weights"] = str(args.checkpoint)
    mc["device"] = "cuda"
    tc = dict(cfg["train_config"]["init_args"])
    for key, enum in (("multi_scale", MultiScale), ("augmentation_backend", AugmentationBackend)):
        if tc.get(key) in enum.__members__:
            tc[key] = enum[tc[key]]
    tc.update(
        devices=1,
        strategy="auto",
        num_workers=0,
        use_ema=False,
        pack_targets=False,
        tensorboard=False,
        progress_bar=None,
        output_dir=str(args.output),
        compute_val_loss=False,
    )
    return RFDETRSegSmallConfig(**mc), SegmentationTrainConfig(**tc)


def coverage(args: argparse.Namespace, mc: Any, tc: Any) -> None:
    """Audit every training image under three independent augmentation seeds."""
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    dataset = data._dataset_train
    assert isinstance(dataset, CocoDetection)
    records: list[dict[str, Any]] = []
    pending: list[Tensor] = []
    metadata: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        short, full, last, extinct = skeleton_trace(torch.stack(pending).cuda())
        counts = short.flatten(1).sum(1).tolist()
        totals = full.flatten(1).sum(1).tolist()
        covered = ((short > 0) & (full > 0)).flatten(1).sum(1).tolist()
        for i, row in enumerate(metadata):
            row.update(
                k10_pixels=counts[i],
                reference_pixels=totals[i],
                coverage=covered[i] / totals[i] if totals[i] else None,
                required_iterations=int(last[i]),
                extinct=bool(extinct[i]),
            )
            records.append(row)
        pending.clear()
        metadata.clear()

    for seed in (42, 43, 44):
        for index in range(len(dataset)):
            seed_sample(seed, index)
            _, target = dataset[index]
            for ti in torch.where(target["labels"] == 1)[0].tolist():
                mask = target["masks"][ti]
                metadata.append(
                    dict(
                        seed=seed,
                        dataset_index=index,
                        image_id=int(target["image_id"].item()),
                        target_index=ti,
                        area=int(mask.sum()),
                        shape=list(mask.shape),
                    )
                )
                pending.append(mask)
                if len(pending) == 24:
                    flush()
            if (index + 1) % 100 == 0:
                print(f"coverage seed={seed} images={index + 1}/{len(dataset)}", flush=True)
        flush()
        write_json(args.output / f"coverage_seed{seed}.json", [r for r in records if r["seed"] == seed])
    nonempty = [r for r in records if r["area"] > 0]
    resolved = [r for r in nonempty if r["extinct"] and r["coverage"] is not None]
    summary = dict(
        images=len(dataset),
        seeds=[42, 43, 44],
        augmented_stems=len(records),
        nonempty_stems=len(nonempty),
        empty_after_augmentation=len(records) - len(nonempty),
        unresolved=sum(not r["extinct"] for r in nonempty),
        empty_k10=sum(r["k10_pixels"] == 0 for r in nonempty),
        coverage_mean=float(np.mean([r["coverage"] for r in resolved])),
        below_95=sum(r["coverage"] < 0.95 for r in resolved),
        below_50=sum(r["coverage"] < 0.5 for r in resolved),
        required_percentiles=np.percentile(
            [r["required_iterations"] for r in resolved], [50, 90, 95, 99, 100]
        ).tolist(),
        transform=repr(dataset._transforms),
    )
    write_json(args.output / "summary.json", summary)
    worst = sorted(resolved, key=lambda r: r["coverage"])[:12]
    for rank, row in enumerate(worst):
        seed_sample(row["seed"], row["dataset_index"])
        sample, target = dataset[row["dataset_index"]]
        mask = target["masks"][row["target_index"]].unsqueeze(0).cuda()
        short, full, _, _ = skeleton_trace(mask)
        rgb = sample * sample.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        rgb = rgb + sample.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        array = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        array[full[0].cpu().bool().numpy()] = (0, 255, 255)
        array[short[0].cpu().bool().numpy()] = (255, 0, 0)
        Image.fromarray(array).save(args.output / f"skeleton_{rank:02d}.png")
    write_json(args.output / "worst.json", worst)


def group_name(name: str) -> str:
    """Partition all trainable parameters into disjoint model-owned groups."""
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("segmentation_head."):
        return "mask_head"
    if "class_embed" in name:
        return "classification"
    if "bbox_embed" in name:
        return "box_head"
    if name.startswith("transformer."):
        return "transformer"
    return "other_shared"


def gradients(args: argparse.Namespace, mc: Any, tc: Any) -> None:
    """Probe weighted original/topology gradients on identical real augmented batches."""
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    dataset = data._dataset_train
    assert isinstance(dataset, CocoDetection)
    collate = make_collate_fn(block_size=mc.patch_size * mc.num_windows, pack=False)
    named = [(name, p) for name, p in module.model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = {
        name: [i for i, (key, _) in enumerate(named) if group_name(key) == name]
        for name in sorted({group_name(key) for key, _ in named})
    }
    groups["all"] = list(range(len(named)))
    write_json(args.output / "parameter_groups.json", {key: [named[i][0] for i in ids] for key, ids in groups.items()})
    order = np.random.default_rng(20261112).permutation(len(dataset)).tolist()
    rows = []
    # Fixed extremes and midpoint are a deliberate scale stress probe, not epoch-frequency weighting.
    for bi in range(args.batches):
        ids = order[bi * 8 : (bi + 1) * 8]
        examples = []
        for index in ids:
            seed_sample(42, index)
            examples.append(dataset[index])
        samples, targets = collate(examples)
        samples = samples.to("cuda")
        targets = [{k: v.cuda() if isinstance(v, Tensor) else v for k, v in t.items()} for t in targets]
        padding = samples.mask
        scale = (264, 384, 504)[bi % 3]
        samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
        samples.mask = F.interpolate(samples.mask[:, None].float(), size=scale, mode="nearest")[:, 0].bool()
        torch.manual_seed(7000 + bi)
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module.model(samples, targets)
            outputs["mask_padding"] = padding
            losses = module.criterion(outputs, targets)
            weighted = {
                k: v * module.criterion.weight_dict[k] for k, v in losses.items() if k in module.criterion.weight_dict
            }
            original = torch.stack([v for k, v in weighted.items() if k != "loss_mask_cldice"]).sum()
            topology = weighted["loss_mask_cldice"]
            region = torch.stack(
                [v for k, v in weighted.items() if k.startswith(("loss_mask_ce", "loss_mask_dice"))]
            ).sum()
        grad_original = list(torch.autograd.grad(original, params, retain_graph=True, allow_unused=True))
        grad_region = list(torch.autograd.grad(region, params, retain_graph=True, allow_unused=True))
        grad_topology = list(torch.autograd.grad(topology, params, allow_unused=True))
        stats = {}
        for name, indices in groups.items():

            def select(values: Sequence[Tensor | None]) -> list[Tensor | None]:
                return [values[i] for i in indices]

            stats[name] = dict(
                original_vs_topology=gradient_relation(select(grad_original), select(grad_topology)),
                region_vs_topology=gradient_relation(select(grad_region), select(grad_topology)),
            )
        torch.cuda.synchronize()
        row = dict(
            batch=bi,
            dataset_indices=ids,
            scale=scale,
            batch_size=len(targets),
            stem_instances=sum(int((t["labels"] == 1).sum()) for t in targets),
            group_detr=mc.group_detr,
            original_loss=float(original),
            weighted_cldice=float(topology),
            region_loss=float(region),
            groups=stats,
            clip_factor=min(1.0, tc.clip_max_norm / max(stats["all"]["original_vs_topology"]["sum_norm"], 1e-12)),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            seconds=time.monotonic() - begin,
        )
        rows.append(row)
        write_json(args.output / "gradients.json", rows)
        print(
            json.dumps({k: row[k] for k in ("batch", "scale", "stem_instances", "weighted_cldice", "seconds")}),
            flush=True,
        )
        del grad_original, grad_region, grad_topology, outputs, losses, weighted, original, topology, region
    assert all(p.grad is None for p in params), "Diagnostic must not populate optimizer gradients"


@torch.no_grad()
def misses(args: argparse.Namespace, mc: Any, tc: Any) -> None:
    """Inspect every query before class-score filtering on the full validation set."""
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().eval()
    tc.batch_size = 1
    tc.eval_batch_size = 1
    data = RFDETRDataModule(mc, tc)
    data.setup("validate")
    converter = COCOEvalCallback(max_dets=500, segmentation=True, eval_backend="vernier")
    rows: list[dict[str, Any]] = []
    thresholds = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    totals = {str(t): dict(tp=0, predictions=0) for t in thresholds}
    for index, (samples, targets) in enumerate(data.val_dataloader()):
        samples = samples.to("cuda")
        targets = [{k: v.cuda() if isinstance(v, Tensor) else v for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw = module.model(samples)
            processed = module.postprocess(raw, torch.stack([t["orig_size"] for t in targets]))
        converted = converter._convert_targets(targets, converter._convert_preds(processed))
        target = converted[0]
        gi = torch.where(target["labels"] == 1)[0]
        logits = raw["pred_logits"][0]
        scores = logits.sigmoid()
        ranks = torch.empty(scores.numel(), device=scores.device, dtype=torch.long)
        ranks[torch.argsort(scores.flatten(), descending=True, stable=True)] = torch.arange(
            scores.numel(), device=scores.device
        )
        stem_ranks = ranks.view_as(scores)[:, 1]
        selected = stem_ranks < module.postprocess.num_select
        h, w = targets[0]["orig_size"].tolist()
        masks = torch.cat(
            [
                F.interpolate(chunk[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0
                for chunk in raw["pred_masks"][0].split(16)
            ]
        )
        gt = target["masks"][gi]
        ious = torch.zeros((len(masks), len(gt)), device="cuda")
        for k, mask in enumerate(gt):
            ious[:, k] = (masks & mask).flatten(1).sum(1) / (masks | mask).flatten(1).sum(1).clamp_min(1)
        box_ious, _ = box_iou(
            box_cxcywh_to_xyxy(raw["pred_boxes"][0]).float(), box_cxcywh_to_xyxy(targets[0]["boxes"][gi]).float()
        )
        base_matches = greedy_matches(ious, scores[:, 1], selected, 0.3)
        for threshold in thresholds:
            matched = greedy_matches(ious, scores[:, 1], selected, threshold)
            totals[str(threshold)]["tp"] += len(matched)
            totals[str(threshold)]["predictions"] += int(((scores[:, 1] >= threshold) & selected).sum())
        for k, ti in enumerate(gi.tolist()):
            candidates = torch.where(ious[:, k] >= 0.5)[0]
            best = int(ious[:, k].argmax())

            def candidate(query: int) -> dict[str, Any]:
                return dict(
                    query=query,
                    iou=float(ious[query, k]),
                    box_iou=float(box_ious[query, k]),
                    class_scores=scores[query].float().tolist(),
                    stem_rank=int(stem_ranks[query]) + 1,
                    selected=bool(selected[query]),
                )

            best_scored = int(candidates[scores[candidates, 1].argmax()]) if len(candidates) else None
            if k in base_matches:
                reason = "matched"
            elif len(candidates) == 0:
                reason = "no_query_mask_iou50"
            elif not (scores[candidates, 1] >= 0.3).any():
                reason = "low_stem_confidence"
            elif not ((scores[candidates, 1] >= 0.3) & selected[candidates]).any():
                reason = "topk_excluded"
            else:
                reason = "greedy_assignment_competition"
            rows.append(
                dict(
                    image_id=int(targets[0]["image_id"].item()),
                    target_index=ti,
                    reason=reason,
                    best_mask=candidate(best),
                    best_box=candidate(int(box_ious[:, k].argmax())),
                    best_scored_iou50=candidate(best_scored) if best_scored is not None else None,
                    matched=candidate(base_matches[k]) if k in base_matches else None,
                    queries=len(masks),
                    classes=scores.shape[1],
                    num_select=module.postprocess.num_select,
                )
            )
        if (index + 1) % 20 == 0:
            print(f"misses images={index + 1}", flush=True)
    write_json(args.output / "queries.json", rows)
    write_json(
        args.output / "summary.json",
        dict(
            stems=len(rows),
            thresholds=totals,
            reasons={name: sum(r["reason"] == name for r in rows) for name in sorted({r["reason"] for r in rows})},
        ),
    )


def main() -> None:
    """Run one bounded diagnostic stage and save source/weight provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("coverage", "gradients", "misses"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()
    if args.stage != "coverage" and args.checkpoint is None:
        parser.error("--checkpoint is required for gradients and misses")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = Path(rfdetr.__file__).parent
    paths = [
        Path(__file__),
        args.config,
        source / "models/criterion.py",
        source / "models/soft_cldice.py",
        source / "training/module_model.py",
        source / "datasets/coco.py",
    ]
    if args.checkpoint:
        paths.append(args.checkpoint)
    write_json(
        args.output / "provenance.json",
        dict(
            stage=args.stage,
            imported_source=str(source),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            optimizer_steps=0,
            arguments=vars(args)
            | {"config": str(args.config), "checkpoint": str(args.checkpoint), "output": str(args.output)},
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        ),
    )
    mc, tc = configurations(args)
    begin = time.monotonic()
    {"coverage": coverage, "gradients": gradients, "misses": misses}[args.stage](args, mc, tc)
    write_json(args.output / "completed.json", dict(complete=True, seconds=time.monotonic() - begin, optimizer_steps=0))


if __name__ == "__main__":
    main()
