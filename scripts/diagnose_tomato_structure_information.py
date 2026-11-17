# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training-split frozen geometry information probe; no optimizer, backward, or loss modification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from scipy.ndimage import label  # type: ignore[import-untyped]
from skimage.morphology import skeletonize  # type: ignore[import-not-found]

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import make_collate_fn
from scripts.diagnose_cldice_dedup import parameter_hash
from scripts.diagnose_tomato_e1 import configurations, seed_sample, write_json

ARMS = ("baseline", "centerline_ce", "centerline_local_ce")


def geometry(
    prediction: NDArray[np.bool_],
    truth: NDArray[np.bool_],
    truth_skeleton: NDArray[np.bool_] | None = None,
    prediction_skeleton: NDArray[np.bool_] | None = None,
) -> dict[str, float | int]:
    """Measure overlap, hard-clDice, overgrowth and connectivity inside each GT component.

    Connectivity counts use connected components, not clDice thresholds. Empty predictions have zero quality.
    """
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    area = int(truth.sum())
    if not area:
        raise ValueError("Geometry requires nonempty GT")
    gs = skeletonize(truth) if truth_skeleton is None else truth_skeleton
    ps = skeletonize(prediction) if prediction_skeleton is None else prediction_skeleton
    inter = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    recall = float((prediction & gs).sum() / max(int(gs.sum()), 1))
    precision = float((ps & truth).sum() / max(int(ps.sum()), 1))
    inside, _ = label(prediction & truth, structure=np.ones((3, 3), dtype=int))
    components, count = label(truth, structure=np.ones((3, 3), dtype=int))
    connected, split = 0, 0
    for component in range(1, count + 1):
        values = inside[gs & (components == component)]
        sizes = np.bincount(values[values > 0])
        connected += int(sizes.max()) if sizes.size else 0
        split += int(np.count_nonzero(sizes) > 1)
    return dict(
        mask_iou=inter / (area + false_positive),
        cldice=2 * recall * precision / (recall + precision) if recall + precision else 0.0,
        center_recall=recall,
        skeleton_precision=precision,
        fp_over_gt=false_positive / area,
        fn_over_gt=(area - inter) / area,
        connected_center_fraction=connected / max(int(gs.sum()), 1),
        split_components=split,
        gt_pixels=area,
        gt_skeleton_pixels=int(gs.sum()),
        pred_pixels=int(prediction.sum()),
    )


def closest_pair(rows: list[dict[str, Any]], caliper: float) -> tuple[int, int] | None:
    """Select one same-GT pair by IoU closeness only, requiring a clDice spread of at least .05.

    Scores never select pairs. IoU must be at least .2; ties prefer larger clDice spread then query order.
    """
    candidates = []
    for i, first in enumerate(rows):
        for j in range(i + 1, len(rows)):
            second = rows[j]
            difference = abs(first["mask_iou"] - second["mask_iou"])
            spread = abs(first["cldice"] - second["cldice"])
            if (
                min(first["mask_iou"], second["mask_iou"]) >= 0.2
                and difference <= caliper
                and spread >= 0.05
                and (first["mask_iou"] >= 0.5) == (second["mask_iou"] >= 0.5)
            ):
                low, high = (i, j) if first["cldice"] < second["cldice"] else (j, i)
                candidates.append((difference, -spread, low, high))
    if not candidates:
        return None
    return min(candidates)[2:]


def dense_masks(outputs: dict[str, Any], canvas: tuple[int, int], size: tuple[int, int]) -> NDArray[np.bool_]:
    """Decode all queries in bounded chunks at the augmented GT grid, excluding padding."""
    masks = outputs["pred_masks"]
    pieces = []
    for start in range(0, outputs["pred_logits"].shape[1], 16):
        if isinstance(masks, torch.Tensor):
            values = masks[0, start : start + 16]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = torch.einsum(
                    "chw,nc->nhw", masks["spatial_features"][0], masks["query_features"][0, start : start + 16]
                )
                values = values + masks["bias"]
        values = F.interpolate(values.float()[:, None], size=canvas, mode="bilinear", align_corners=False)
        pieces.append((values[:, 0, : size[0], : size[1]] > 0).cpu().numpy())
    return np.concatenate(pieces)


def select_images(dataset: CocoDetection) -> list[dict[str, Any]]:
    """Preselect 128 stem and 32 non-stem training images and disjoint discovery/confirmation halves."""
    assert dataset.cat2label is not None
    categories = {c for c, v in dataset.cat2label.items() if v == 1}
    pools: dict[str, list[int]] = {"stem": [], "background": []}
    for i, image_id in enumerate(dataset.ids):
        role = "stem" if any(a["category_id"] in categories for a in dataset.coco.imgToAnns[image_id]) else "background"
        pools[role].append(i)
    rng = np.random.default_rng(2026111701)
    selected = []
    for role, number in (("stem", 128), ("background", 32)):
        assert len(pools[role]) >= number
        for k, i in enumerate(rng.choice(pools[role], number, replace=False).tolist()):
            selected.append(
                dict(
                    dataset_index=i,
                    image_id=dataset.ids[i],
                    role=role,
                    cohort="discovery" if k < number // 2 else "confirmation",
                )
            )
    return selected


def summarize(records: list[dict[str, Any]], output: Path) -> None:
    """Retain per-GT conditional pairs and focus counts; no classifier fitting or performance claims."""
    pairs, summaries = [], []
    for arm in ARMS:
        for cohort in ("discovery", "confirmation"):
            cases = [g for r in records if r["arm"] == arm and r["cohort"] == cohort for g in r["stems"]]
            summaries.append(
                dict(
                    arm=arm,
                    cohort=cohort,
                    gt_observations=len(cases),
                    low_score_good=sum(g["low_score_good"] for g in cases),
                    center_overgrowth=sum(
                        any(c["center_recall"] >= 0.9 and c["fp_over_gt"] >= 0.5 for c in g["candidates"])
                        for g in cases
                    ),
                    cldice_overgrowth=sum(
                        any(c["cldice"] >= 0.9 and c["fp_over_gt"] >= 0.5 for c in g["candidates"]) for g in cases
                    ),
                )
            )
    for record in records:
        for gt in record["stems"]:
            for caliper in (0.02, 0.005):
                pair = closest_pair(gt["candidates"], caliper)
                if pair is None:
                    continue
                low, high = [gt["candidates"][i] for i in pair]
                keys = (
                    "mask_iou",
                    "cldice",
                    "score",
                    "fp_over_gt",
                    "fn_over_gt",
                    "connected_center_fraction",
                    "split_components",
                )
                pairs.append(
                    dict(
                        arm=record["arm"],
                        cohort=record["cohort"],
                        image_id=record["image_id"],
                        augmentation_seed=record["augmentation_seed"],
                        target_index=gt["target_index"],
                        caliper=caliper,
                        low=low,
                        high=high,
                        differences={key: high[key] - low[key] for key in keys},
                    )
                )
    write_json(output / "conditional-pairs.json", pairs)
    write_json(output / "focus-summary.json", summaries)


def run(short_run: Path, output: Path, expected_commit: str) -> None:
    """Inspect three fixed epoch-ten checkpoints on training data; assert weights and input audits stay unchanged."""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    completed = json.loads((short_run / "completed.json").read_text())
    assert completed["complete"]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    reference_audit: list[dict[str, Any]] = []
    provenance = dict(
        git_commit=expected_commit,
        checkpoints=completed["checkpoint_sha256"],
        split="train",
        selection_seed=2026111701,
        augmentation_seeds=[42, 43],
        mode="eval on native augmented training images, batch1, 100 queries",
        stem_images=128,
        background_images=32,
        cohorts="disjoint image halves, not independent augmentations or queries",
        candidate_selection="IoU>=.1 OR GT-center-recall>=.9 OR native matched OR highest stem score OR best IoU",
        pair_selection=(
            "same GT/arm/augmentation; IoU>=.2 and same IoU50 class; clDice spread>=.05; "
            "closest IoU then largest clDice spread; no scores"
        ),
        primary_iou_caliper=0.02,
        sensitivity_iou_caliper=0.005,
        low_score=0.3,
        good_iou=0.5,
        high_center=0.9,
        severe_fp_over_gt=0.5,
        high_cldice=0.9,
        limits=(
            "hard-clDice geometry information only; no validation use; "
            "not a classifier, gradient or recall improvement test"
        ),
    )
    for arm in ARMS:
        checkpoint = short_run / arm / "last_ema.pth"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == completed["checkpoint_sha256"][arm]
        mc, tc = configurations(
            argparse.Namespace(config=short_run / f"{arm}.yaml", checkpoint=checkpoint, output=output)
        )
        tc.amp_dtype = None
        module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().eval()
        module.requires_grad_(False)
        before = parameter_hash(module.model)
        data = RFDETRDataModule(mc, tc)
        data.setup("fit")
        dataset = data._dataset_train
        assert isinstance(dataset, CocoDetection)
        if not selected:
            selected = select_images(dataset)
            assert tc.dataset_dir is not None
            annotation = Path(tc.dataset_dir) / "train/_annotations.coco.json"
            provenance.update(
                selection=selected,
                train_annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                transforms=repr(dataset._transforms),
            )
            write_json(output / "protocol.json", provenance)
        collate = make_collate_fn(block_size=mc.patch_size * mc.num_windows, pack=False)
        arm_records: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        examples: dict[str, list[int]] = {}
        with torch.inference_mode():
            for seed in (42, 43):
                for position, chosen in enumerate(selected):
                    index = chosen["dataset_index"]
                    seed_sample(seed, index)
                    samples, targets = collate([dataset[index]])
                    target = targets[0]
                    input_digest = hashlib.sha256(
                        samples.tensors.numpy().tobytes()
                        + target["masks"].numpy().tobytes()
                        + target["labels"].numpy().tobytes()
                    ).hexdigest()
                    audit.append(
                        dict(
                            image_id=chosen["image_id"],
                            seed=seed,
                            sha256=input_digest,
                            shape=list(samples.tensors.shape),
                        )
                    )
                    target = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = module.model(samples.to("cuda"))
                        _, native_labels, native_queries = module.postprocess._select_topk(outputs["pred_logits"])
                        indices = module.criterion.matcher(outputs, [target], group_detr=1)[0]
                    assert outputs["pred_logits"].shape[1] == 100
                    scores = outputs["pred_logits"][0, :, 1].sigmoid().float().cpu().numpy()
                    native = set(native_queries[0][native_labels[0] == 1].tolist())
                    matches = {int(t): int(q) for q, t in zip(*indices, strict=True)}
                    stem_ids = torch.where(target["labels"] == 1)[0].tolist()
                    record = dict(
                        arm=arm,
                        **chosen,
                        augmentation_seed=seed,
                        input_shape=list(samples.tensors.shape),
                        native_stems=[dict(query=q, score=float(scores[q])) for q in sorted(native)],
                        stems=[],
                    )
                    if stem_ids:
                        truth_masks = target["masks"].cpu().numpy().astype(bool)
                        predicted = dense_masks(
                            outputs, tuple(samples.tensors.shape[-2:]), tuple(truth_masks.shape[-2:])
                        )
                        skeleton_cache = {}
                        for ti in stem_ids:
                            truth = truth_masks[ti]
                            if not truth.any():
                                continue
                            gs = skeletonize(truth)
                            intersection = (predicted & truth).sum((1, 2))
                            ious = intersection / np.maximum((predicted | truth).sum((1, 2)), 1)
                            center = (predicted & gs).sum((1, 2)) / max(int(gs.sum()), 1)
                            good = np.flatnonzero(ious >= 0.5)
                            selected_queries = set(np.flatnonzero((ious >= 0.1) | (center >= 0.9)).tolist())
                            selected_queries.update((int(scores.argmax()), int(ious.argmax()), matches[ti]))
                            candidates = []
                            for q in sorted(selected_queries):
                                if q not in skeleton_cache:
                                    skeleton_cache[q] = skeletonize(predicted[q])
                                metric = geometry(predicted[q], truth, gs, skeleton_cache[q])
                                assert abs(metric["mask_iou"] - float(ious[q])) < 1e-10
                                candidates.append(
                                    dict(
                                        query=q,
                                        score=float(scores[q]),
                                        native=q in native,
                                        matched=q == matches[ti],
                                        **metric,
                                    )
                                )
                            low = bool(len(good) and float(scores[good].max()) < 0.3)
                            record["stems"].append(
                                dict(target_index=ti, good_queries=len(good), low_score_good=low, candidates=candidates)
                            )
                            focus = {}
                            if low:
                                focus["low_score_good"] = max(
                                    (c for c in candidates if c["mask_iou"] >= 0.5), key=lambda c: c["score"]
                                )
                            for name, key in (("center_overgrowth", "center_recall"), ("cldice_overgrowth", "cldice")):
                                eligible = [c for c in candidates if c[key] >= 0.9 and c["fp_over_gt"] >= 0.5]
                                if eligible:
                                    focus[name] = max(eligible, key=lambda c: c["fp_over_gt"])
                            for name, candidate in focus.items():
                                seen = examples.setdefault(name, [])
                                if len(seen) >= 3 or chosen["image_id"] in seen:
                                    continue
                                seen.append(chosen["image_id"])
                                q = candidate["query"]
                                example = output / f"{arm}-{name}-{len(seen)}.npz"
                                normalized = (
                                    samples.tensors[0, :, : truth.shape[0], : truth.shape[1]]
                                    .cpu()
                                    .numpy()
                                    .transpose(1, 2, 0)
                                )
                                rgb = np.clip(
                                    (normalized * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255
                                ).astype(np.uint8)
                                np.savez_compressed(
                                    example,
                                    rgb=rgb,
                                    gt=truth,
                                    prediction=predicted[q],
                                    gt_skeleton=gs,
                                    prediction_skeleton=skeleton_cache[q],
                                )
                                write_json(
                                    example.with_suffix(".json"),
                                    dict(
                                        arm=arm,
                                        **chosen,
                                        augmentation_seed=seed,
                                        target_index=ti,
                                        focus=name,
                                        **candidate,
                                    ),
                                )
                    arm_records.append(record)
                    if (position + 1) % 16 == 0:
                        print(f"STRUCTURE_INFO arm={arm} seed={seed} images={position + 1}/{len(selected)}", flush=True)
        if not reference_audit:
            reference_audit = audit
        assert audit == reference_audit
        assert before == parameter_hash(module.model)
        assert all(p.grad is None for p in module.parameters())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == completed["checkpoint_sha256"][arm]
        write_json(output / f"{arm}-records.json", arm_records)
        write_json(output / f"{arm}-input-audit.json", audit)
        records.extend(arm_records)
        del module, data, outputs
        gc.collect()
        torch.cuda.empty_cache()
    summarize(records, output)
    write_json(
        output / "completed.json",
        dict(
            complete=True,
            checkpoint_hashes_unchanged=True,
            parameters_unchanged=True,
            optimizer_steps=0,
            backward_calls=0,
            parameter_gradients=False,
            input_audits_equal=True,
            elapsed_seconds=time.perf_counter() - started,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        ),
    )
    print("STRUCTURE_INFORMATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.short_run, args.output, args.expected_commit)
