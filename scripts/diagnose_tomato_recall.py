# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Diagnose stem recall across scores and all queries without updating model weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils

from rfdetr.models.postprocess import PostProcess
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.box_ops import box_iou
from scripts.diagnose_tomato_e1 import configurations, write_json


def recall_curve(records: list[dict[str, Any]], iou_key: str = "mask_iou") -> list[dict[str, Any]]:
    """Sweep every distinct native stem score with score-ordered one-to-one matching.

    Equal scores enter together; native postprocessing order breaks within-image ties. Images without stem GT still
    contribute false positives. All-query candidates are diagnostic only and never change the native PR curve.
    """
    total = sum(len(row["target_indices"]) for row in records)
    candidates = sorted(
        (-p["score"], i, rank, p["query"])
        for i, row in enumerate(records)
        for rank, p in enumerate(row["native_stems"])
    )
    used: list[set[int]] = [set() for _ in records]
    tp = 0
    points = []
    for n, (negative_score, i, _, query) in enumerate(candidates, start=1):
        available = [k for k in range(len(records[i]["target_indices"])) if k not in used[i]]
        if available:
            k = max(available, key=lambda k: records[i][iou_key][query][k])
            if records[i][iou_key][query][k] >= 0.5:
                used[i].add(k)
                tp += 1
        if n == len(candidates) or candidates[n][0] != negative_score:
            points.append(
                dict(
                    threshold=-negative_score,
                    tp=tp,
                    fp=n - tp,
                    misses=total - tp,
                    precision=tp / n,
                    recall=tp / max(1, total),
                )
            )
    return points


def at_threshold(curve: list[dict[str, Any]], threshold: float, total: int) -> dict[str, Any]:
    """Read an attainable operating point with score >= threshold, including an empty prediction set."""
    points = [p for p in curve if p["threshold"] >= threshold]
    return (
        dict(points[-1], threshold=threshold)
        if points
        else dict(threshold=threshold, tp=0, fp=0, misses=total, precision=0.0, recall=0.0)
    )


def diagnose_misses(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Separate threshold/top-k failures from mask and box coverage at fixed confidence 0.3.

    A low-score geometric candidate is a diagnostic possibility, not a verified detection. Other-class competition is
    recorded independently because its presence does not establish the cause of the miss.
    """
    candidates = [p for p in row["native_stems"] if p["score"] >= 0.3]
    matched = set()
    for p in candidates:
        available = [k for k in range(len(row["target_indices"])) if k not in matched]
        if available:
            k = max(available, key=lambda k: row["mask_iou"][p["query"]][k])
            if row["mask_iou"][p["query"]][k] >= 0.5:
                matched.add(k)
    masks, boxes, scores = (np.asarray(row[k]) for k in ("mask_iou", "box_iou", "query_scores"))
    misses = []
    for k, target_index in enumerate(row["target_indices"]):
        if k in matched:
            continue
        fit = np.flatnonzero(masks[:, k] >= 0.5)
        high = [q for q in fit if scores[q, 1] >= 0.3]
        if high:
            reason = "assignment_conflict" if any(p["query"] in high for p in candidates) else "native_topk_filter"
        elif len(fit):
            reason = "low_stem_score"
        elif boxes[:, k].max() >= 0.5:
            reason = "box_found_mask_below_iou50"
        else:
            reason = "no_query_mask_or_box_iou50"
        query = int(max(fit, key=lambda q: scores[q, 1])) if len(fit) else int(masks[:, k].argmax())
        other = max(float(scores[query, c]) for c in (0, 2))
        misses.append(
            dict(
                image_id=row["image_id"],
                target_index=target_index,
                reason=reason,
                query=query,
                mask_iou=float(masks[query, k]),
                box_iou=float(boxes[query, k]),
                stem_score=float(scores[query, 1]),
                other_class_score=other,
                other_class_competition=bool(len(fit) and other >= 0.3 and other > scores[query, 1]),
            )
        )
    return misses


def evaluate(config: Path, checkpoint: Path, output: Path, previous: Path) -> dict[str, Any]:
    """Run identical validation preprocessing, retaining every query's geometry and class scores."""
    output.mkdir(exist_ok=False)
    mc, tc = configurations(argparse.Namespace(config=config, checkpoint=checkpoint, output=output))
    tc.batch_size = tc.eval_batch_size = 1
    tc.amp_dtype = None
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().eval()
    data = RFDETRDataModule(mc, tc)
    data.setup("validate")
    converter = COCOEvalCallback(max_dets=500, segmentation=True, eval_backend="vernier")
    full = PostProcess(num_select=mc.num_queries)
    records = []
    with torch.inference_mode():
        for index, (samples, targets) in enumerate(data.val_dataloader()):
            samples = samples.to("cuda")
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            sizes = torch.stack([t["orig_size"] for t in targets])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw = module.model(samples)
                native = module.postprocess(raw, sizes)
                # One class keeps one mask per query, including queries outside the native top-k.
                all_masks = full({**raw, "pred_logits": raw["pred_logits"][..., 1:2]}, sizes)
            preds = converter._convert_preds(native)
            truth = converter._convert_targets(targets, preds)[0]
            gi = torch.where(truth["labels"] == 1)[0]
            _, labels, queries = module.postprocess._select_topk(raw["pred_logits"])
            _, _, all_order = full._select_topk(raw["pred_logits"][..., 1:2])
            reverse = torch.argsort(all_order[0])
            pred_masks = all_masks[0]["masks"][reverse, 0].cpu().numpy().astype(np.uint8)
            gt_masks = truth["masks"][gi].cpu().numpy().astype(np.uint8)
            if len(gi):
                pred_rle = mask_utils.encode(np.asfortranarray(pred_masks.transpose(1, 2, 0)))
                gt_rle = mask_utils.encode(np.asfortranarray(gt_masks.transpose(1, 2, 0)))
                overlaps = mask_utils.iou(pred_rle, gt_rle, [0] * len(gi)).tolist()
                box_overlaps = box_iou(all_masks[0]["boxes"][reverse].float(), truth["boxes"][gi].float())[0].tolist()
            else:
                overlaps = [[] for _ in range(mc.num_queries)]
                box_overlaps = [[] for _ in range(mc.num_queries)]
            image_id = int(targets[0]["image_id"].item())
            record = dict(
                image_id=image_id,
                target_indices=gi.tolist(),
                query_scores=raw["pred_logits"][0].sigmoid().tolist(),
                mask_iou=overlaps,
                box_iou=box_overlaps,
                native_stems=[
                    dict(query=int(q), score=float(s))
                    for q, c, s in zip(queries[0], labels[0], native[0]["scores"], strict=True)
                    if int(c) == 1
                ],
            )
            records.append(record)
            if (index + 1) % 20 == 0:
                print(json.dumps(dict(arm=output.name, images=index + 1)), flush=True)
    curve = recall_curve(records)
    total = sum(len(row["target_indices"]) for row in records)
    fixed = at_threshold(curve, 0.3, total)
    prior = json.loads((previous / "summary.json").read_text())
    assert len(records) == prior["images"] == 199 and total == prior["stem_gt"] == 121
    assert fixed["tp"] == prior["stem_true_positives"] and fixed["tp"] + fixed["fp"] == prior["stem_predictions"]
    old_instances = json.loads((previous / "stem_instances.json").read_text())
    misses = [m for row in records for m in diagnose_misses(row)]
    assert {(m["image_id"], m["target_index"]) for m in misses} == {
        (r["image_id"], r["target_index"]) for r in old_instances if not r["matched"]
    }
    summary = dict(
        images=len(records),
        gt_stems=total,
        native_num_select=mc.num_select,
        all_queries=mc.num_queries,
        fixed_0p3=fixed,
        reasons=dict(Counter(m["reason"] for m in misses)),
        other_class_competition=sum(m["other_class_competition"] for m in misses),
        mask_operating_points=[at_threshold(curve, t, total) for t in (0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)],
        box_operating_points=[at_threshold(recall_curve(records, "box_iou"), t, total) for t in (0.1, 0.3, 0.5)],
        previous_fixed_operating_point_reproduced=True,
    )
    write_json(output / "queries.json", records)
    write_json(output / "mask_pr_curve.json", curve)
    write_json(output / "misses.json", misses)
    write_json(output / "summary.json", summary)
    del module, data
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    """Evaluate only the existing fixed-epoch checkpoints and preserve provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--arms", nargs=2, default=("baseline", "cldice"))
    parser.add_argument("--precision-floor", type=float, default=None)
    args = parser.parse_args()
    if args.precision_floor is not None and not 0 <= args.precision_floor <= 1:
        parser.error("--precision-floor must be between zero and one")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert json.loads((args.pair_root / "completed.json").read_text())["complete"]
    checkpoints = [args.pair_root / arm / "last_ema.pth" for arm in args.arms]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "provenance.json",
        dict(
            git_commit=args.expected_commit,
            arms=args.arms,
            requested_precision_floor=args.precision_floor,
            checkpoint_sha256=before,
            optimizer_steps=0,
            scope="native top-k PR plus all-query geometry and scores; no training",
        ),
    )
    torch.set_num_threads(2)
    summaries = {}
    for arm, checkpoint in zip(args.arms, checkpoints, strict=True):
        torch.manual_seed(42)
        summaries[arm] = evaluate(
            args.pair_root / f"{arm}.yaml",
            checkpoint,
            args.output / arm,
            args.pair_root / "evaluation" / f"{arm}-last10",
        )
    target_precision = (
        summaries[args.arms[0]]["fixed_0p3"]["precision"] if args.precision_floor is None else args.precision_floor
    )
    comparable = {}
    for arm in summaries:
        curve = json.loads((args.output / arm / "mask_pr_curve.json").read_text())
        choices = [point for point in curve if point["precision"] >= target_precision]
        comparable[arm] = max(choices, key=lambda p: (p["recall"], p["threshold"])) if choices else None
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
    write_json(
        args.output / "completed.json",
        dict(
            complete=True,
            optimizer_steps=0,
            checkpoints_unchanged=True,
            target_precision=target_precision,
            max_recall_at_precision_floor=comparable,
            summaries=summaries,
        ),
    )
    print(json.dumps(dict(complete=True, target_precision=target_precision, comparable=comparable)), flush=True)


if __name__ == "__main__":
    main()
