# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Score tomato picking points on all visible GT points, including missed stem instances.

Run with python -m scripts.evaluate_tomato_points --help. Archived input is one record per image (including empty
images), with image_id, scores and pred_masks (stem-only COCO RLE masks at original resolution), and optionally aligned
pick_points in original xy pixels (null for a missing point). Use the fixed score >= 0.30 policy of the original
baseline prediction archive.
"""

import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as masks
from scipy import ndimage  # type: ignore[import-untyped]
from skimage.morphology import skeletonize  # type: ignore[import-not-found]

Point = tuple[float, float]
Pixel = tuple[int, int]
Record = dict[str, Any]


def farthest(nodes: set[Pixel], start: Pixel) -> tuple[Pixel, dict[Pixel, float], dict[Pixel, Pixel]]:
    """Walk the eight-connected skeleton using pixel arc length and the annotation rule's tie order."""
    distance = {start: 0.0}
    parents: dict[Pixel, Pixel] = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        y, x = node
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbor = y + dy, x + dx
                candidate = cost + math.hypot(dx, dy)
                if neighbor in nodes and candidate < distance.get(neighbor, math.inf):
                    parents[neighbor] = node
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
    return max(distance, key=distance.__getitem__), distance, parents


def geometry(mask: Any) -> dict[str, Point]:
    """Compute an area centroid and the historical largest-component skeleton path midpoint."""
    y, x = np.nonzero(mask)
    if not len(x):
        return {}
    result = {"centroid": (float(x.mean()), float(y.mean()))}
    labels, _ = ndimage.label(mask)
    largest = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    nodes = set(zip(*np.nonzero(skeletonize(labels == largest)), strict=True))
    if nodes:
        start, _, _ = farthest(nodes, min(nodes))
        end, dist, parents = farthest(nodes, start)
        path = [end]
        while path[-1] != start:
            path.append(parents[path[-1]])
        ymid, xmid = min(reversed(path), key=lambda p: abs(dist[p] - dist[end] / 2))
        result["skeleton"] = (int(xmid), int(ymid))
    return result


def evaluate(dataset: Record, predictions: Iterable[Record]) -> tuple[Record, list[Record]]:
    """Match stem masks once by descending score, then score all methods against the same GT instances."""
    images = {im["id"]: im for im in dataset["images"]}
    targets: dict[int, list[Record]] = defaultdict(list)
    for gt in dataset["annotations"]:
        if gt["category_id"] == 1:
            if gt.get("iscrowd", 0):
                raise ValueError("This tomato protocol requires instance annotations without crowd regions")
            targets[gt["image_id"]].append(gt)
    rows: list[Record] = []
    seen: set[int] = set()
    tp = fp = fn = 0
    modes: set[bool] = set()
    for record in predictions:
        image_id = record["image_id"]
        if image_id not in images or image_id in seen:
            raise ValueError("Duplicate or unknown image in prediction coverage")
        seen.add(image_id)
        image = images[image_id]
        shape = [image["height"], image["width"]]
        scores, rles = record["scores"], record["pred_masks"]
        head = record.get("pick_points")
        modes.add(head is not None)
        if len(scores) != len(rles) or (head is not None and len(head) != len(scores)):
            raise ValueError("Per-instance scores, masks and points must be aligned")
        if any(list(r["size"]) != shape for r in rles) or not all(math.isfinite(s) for s in scores):
            raise ValueError("Invalid score or mask canvas; use original image coordinates")
        selected = sorted((j for j, s in enumerate(scores) if s >= 0.30), key=lambda j: -scores[j])
        truth = targets[image_id]
        available = set(range(len(truth)))
        matched = {}
        if selected and truth:
            overlaps = masks.iou([rles[j] for j in selected], [g["segmentation"] for g in truth], [0] * len(truth))
            for idx, j in enumerate(selected):
                if not available:
                    break
                k = max(sorted(available), key=lambda k: overlaps[idx, k])
                if overlaps[idx, k] >= 0.50:
                    matched[k] = j
                    available.remove(k)
        tp += len(matched)
        fp += len(selected) - len(matched)
        fn += len(truth) - len(matched)
        for k, gt in enumerate(truth):
            x, y, visibility = gt["keypoints"]
            if visibility <= 0:
                continue
            diag = math.hypot(*gt["bbox"][2:])
            if not math.isfinite(diag) or diag <= 0:
                raise ValueError("Invalid annotated stem box")
            matched_prediction = matched.get(k)
            row: Record = dict(
                image_id=image_id,
                annotation_id=gt["id"],
                group=gt.get("tomato_keypoint", {}).get("group_id"),
                matched=matched_prediction is not None,
                gt_x=x,
                gt_y=y,
                box_diagonal=diag,
            )
            if matched_prediction is not None:
                points = geometry(masks.decode(rles[matched_prediction]).astype(bool))
                if head is not None and head[matched_prediction] is not None:
                    px, py = head[matched_prediction]
                    if math.isfinite(px) and math.isfinite(py) and 0 <= px < shape[1] and 0 <= py < shape[0]:
                        points["point_head"] = (px, py)
                gt_mask = masks.decode(gt["segmentation"])
                for name, (px, py) in points.items():
                    row[name + "_x"], row[name + "_y"] = px, py
                    row[name + "_error_px"] = math.hypot(px - x, py - y)
                    row[name + "_inside"] = bool(gt_mask[min(round(py), shape[0] - 1), min(round(px), shape[1] - 1)])
            rows.append(row)
    if seen != set(images) or len(modes) != 1:
        raise ValueError("Incomplete image coverage or inconsistent presence of point-head outputs")
    detected = sum(r["matched"] for r in rows)
    result: Record = dict(
        images=len(seen), tp=tp, fp=fp, fn=fn, visible_points=len(rows), detected_visible=detected, methods={}
    )
    for name in ["centroid", "skeleton"] + (["point_head"] if True in modes else []):
        scored = [r for r in rows if name + "_error_px" in r]
        err = np.array([r[name + "_error_px"] for r in scored], dtype=float)
        nme = err / np.array([r["box_diagonal"] for r in scored])
        stats: Record = dict(
            points_returned=len(scored),
            mean_px=float(err.mean()) if len(err) else None,
            median_px=float(np.median(err)) if len(err) else None,
            mean_nme=float(nme.mean()) if len(err) else None,
            inside_gt_mask=sum(r[name + "_inside"] for r in scored),
            within5=int((err <= 5).sum()),
            within10=int((err <= 10).sum()),
        )
        for tol in (0.05, 0.10, 0.20):
            hit = int((nme <= tol).sum())
            stats[f"pck_{tol:.2f}"] = dict(
                hits=hit, conditional=hit / detected if detected else None, end_to_end=hit / len(rows) if rows else None
            )
        result["methods"][name] = stats
    return result, rows


def from_checkpoint(config: Path, checkpoint: Path, annotation: Path, device: str) -> Iterable[Record]:
    """Emit the native top-k masks and optional points from the same query rows on the validation split."""
    import torch

    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from scripts.diagnose_tomato_e1 import configurations

    mc, tc = configurations(
        argparse.Namespace(config=config, checkpoint=checkpoint.resolve(), output=checkpoint.parent)
    )
    if tc.dataset_dir is None:
        raise ValueError("Validation requires a configured dataset directory")
    if (Path(tc.dataset_dir) / "valid" / "_annotations.coco.json").read_bytes() != annotation.read_bytes():
        raise ValueError("The validation annotations must match the configured dataset")
    if mc.query_point_head:
        weights = torch.load(checkpoint, map_location="meta", weights_only=False)
        keys = weights.get("model", weights.get("state_dict", {}))
        if not any("query_point_embed." in k for k in keys):
            raise ValueError("An E2 point evaluation requires a checkpoint containing the trained point head")
        del keys, weights
    mc.device = device
    tc.eval_batch_size = tc.batch_size = 1
    tc.amp_dtype = None
    model = RFDETRModelModule(mc, tc, load_encoder_weights=False).to(device).eval()
    dm = RFDETRDataModule(mc, tc)
    dm.setup("validate")
    for index, (x, ts) in enumerate(dm.val_dataloader()):
        ts = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in ts]
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model.validation_step((x.to(device), ts), index)["results"][0]
        ids = torch.where((pred["labels"] == 1) & (pred["scores"] >= 0.30))[0]
        seg = pred["masks"][ids].reshape(-1, *pred["masks"].shape[-2:]).cpu().numpy().astype(np.uint8)
        row: Record = dict(
            image_id=int(ts[0]["image_id"].item()),
            scores=pred["scores"][ids].float().tolist(),
            pred_masks=[masks.encode(np.asfortranarray(m)) for m in seg],
        )
        if mc.query_point_head:
            row["pick_points"] = [
                xy if v else None
                for xy, v in zip(
                    pred["pick_points"][ids].float().tolist(), pred["pick_point_valid"][ids].tolist(), strict=True
                )
            ]
        yield row


def main() -> None:
    """Write a per-instance CSV and aggregate counts for archive replay or direct checkpoint inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Model and dataset YAML for checkpoint evaluation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True, help="New directory for summary.json and instances.csv")
    args = parser.parse_args()
    if args.checkpoint and not args.config:
        parser.error("--checkpoint requires --config")
    predictions = (
        json.loads(args.predictions.read_bytes())
        if args.predictions
        else from_checkpoint(args.config, args.checkpoint, args.annotations, args.device)
    )
    result, rows = evaluate(json.loads(args.annotations.read_bytes()), predictions)
    source_file = args.predictions or args.checkpoint
    result.update(
        score_threshold=0.30,
        mask_iou_threshold=0.50,
        source=source_file.name,
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "instances.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
