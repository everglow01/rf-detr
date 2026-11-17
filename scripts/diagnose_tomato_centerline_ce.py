# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Frozen GT-centerline CE comparison and training-only coefficient calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from scripts.diagnose_tomato_decoder_gradients import choose_cohort, probe_batch
from scripts.diagnose_tomato_e1 import configurations, seed_sample, write_json

CE_MODES = ("centerline_ce", "centerline_local_ce")


def random_training_cohort(dataset: CocoDetection, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select sixteen distinct augmented training images without inspecting predictions or validation data."""
    rng = np.random.default_rng(20261117)
    excluded = {row["image_id"] for row in existing}
    result: list[dict[str, Any]] = []
    for index in rng.permutation(len(dataset)).tolist():
        seed_sample(42, index)
        _, target = dataset[index]
        image_id = int(target["image_id"].item())
        stems = torch.where((target["labels"] == 1) & target["masks"].flatten(1).any(1))[0].tolist()
        if image_id in excluded or not stems:
            continue
        chosen = int(rng.choice(stems))
        result.append(
            dict(
                pair=8 + len(result),
                role="random",
                augmentation_seed=42,
                dataset_index=index,
                image_id=image_id,
                target_index=chosen,
                gt_mask_pixels=int(target["masks"][chosen].sum()),
            )
        )
        excluded.add(image_id)
        if len(result) == 16:
            return result
    raise RuntimeError("Could not select sixteen distinct nonempty stem training images")


def choose_coefficient(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fix one B/C coefficient at 10% median random-baseline scale, capped at unit ratio on observed extremes.

    Ratios in rows were measured with outer coefficient 0.1. No gradient angle or validation score selects the weight.
    """
    calibration = [
        r["norm_ratio"]
        for r in rows
        if r["arm"] == "baseline" and r["role"] == "random" and r["mode"] == "centerline_local_ce"
    ]
    assert len(calibration) == 16 and all(np.isfinite(v) and v >= 0 for v in calibration)
    typical = median(calibration)
    maximum = max(r["max_seed_norm_ratio"] for r in rows if r["mode"] in CE_MODES)
    assert typical > 0 and maximum > 0
    uncapped = 0.01 / typical
    coefficient = min(uncapped, 0.1 / maximum)
    return dict(
        coefficient=coefficient,
        uncapped_coefficient=uncapped,
        extreme_cap_applied=coefficient < uncapped,
        target_median_ratio=0.1,
        observed_random_median_ratio=typical * coefficient / 0.1,
        observed_maximum_ratio=maximum * coefficient / 0.1,
        rule="C random-baseline median=10%; same B/C coefficient; observed per-GT per-seed ratio capped at 1",
    )


def summarize(output: Path, cohort: list[dict[str, Any]]) -> None:
    """Keep per-GT directions and cost evidence; freeze a single common coefficient before any validation."""
    rows, performance = [], []
    for arm in ("baseline", "beta1", "beta2"):
        for batch in range(4):
            data = json.loads((output / arm / f"batch{batch}.json").read_text())
            arrays = np.load(output / arm / f"batch{batch}-gradients.npz", allow_pickle=False)
            assert data["forward_bitwise_equal"] and not data["parameter_gradients"]
            assert data["weight_dict"]["loss_mask_ce"] == data["weight_dict"]["loss_mask_dice"] == 5
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
            for mode in ("gt_centerline", *CE_MODES):
                points = [p for p in data["performance"] if p["variant"] == mode]
                assert len(points) == 5
                performance.append(
                    dict(
                        arm=arm,
                        batch=batch,
                        mode=mode,
                        **{
                            k: median(p[k] for p in points) for k in ("seconds", "peak_allocated_mib", "extra_peak_mib")
                        },
                    )
                )
            for case in cohort[batch * 8 : batch * 8 + 8]:
                ids = [
                    i for i, m in enumerate(data["metadata"]) if (m["pair"], m["role"]) == (case["pair"], case["role"])
                ]
                assert len(ids) == 13
                gradients = {k: arrays[k][ids].astype(np.float64).ravel() for k in arrays.files}
                native = [gradients[f"mask_{seed}"] for seed in (420, 421, 422)]
                assert all(np.linalg.norm(g) > 0 for g in native)
                for mode in ("gt_centerline", *CE_MODES):
                    extra = gradients[mode]
                    norm = float(np.linalg.norm(extra))
                    ratios = [norm / float(np.linalg.norm(g)) for g in native]
                    cosines = {}
                    for key in ("dense_center_missing", "dense_false_positive", "cls"):
                        objective = gradients[key]
                        denominator = norm * float(np.linalg.norm(objective))
                        cosines[key] = (
                            float(np.clip(np.dot(extra, objective) / denominator, -1, 1)) if denominator else None
                        )
                    rows.append(
                        dict(
                            arm=arm,
                            batch=batch,
                            **case,
                            mode=mode,
                            norm_ratio=median(ratios),
                            max_seed_norm_ratio=max(ratios),
                            norm=norm,
                            cosines=cosines,
                        )
                    )
    selection = choose_coefficient(rows)
    write_json(output / "per-gt.json", rows)
    write_json(output / "performance.json", performance)
    write_json(
        output / "calibration.json",
        dict(
            **selection,
            native_weights=dict(ce=5, dice=5, classification=1),
            radius=3,
            internal_negative_weight=1,
            finite=True,
            optimizer_steps=0,
        ),
    )
    print("CE_CALIBRATION " + json.dumps(selection), flush=True)


def run(previous: Path, output: Path, expected_commit: str) -> None:
    """Run the fixed difficult/control cohort plus randomly selected training GTs on three frozen checkpoints."""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert json.loads((previous / "completed.json").read_text())["complete"]
    output.mkdir(parents=True, exist_ok=False)
    cohort = choose_cohort(previous)
    checkpoints = json.loads((previous / "protocol.json").read_text())["checkpoints"]
    torch.set_num_threads(2)
    for row in checkpoints:
        checkpoint = Path(row["path"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
        target = output / row["arm"]
        target.mkdir()
        config = checkpoint.parent.parent / ("training.yaml" if row["arm"] == "baseline" else f"{row['arm']}.yaml")
        mc, tc = configurations(argparse.Namespace(config=config, checkpoint=checkpoint, output=target))
        tc.amp_dtype = None
        module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
        module.requires_grad_(False)
        criterion = module.criterion
        criterion.mask_cldice_class_ids = (1,)
        criterion.mask_cldice_iterations, criterion.mask_cldice_chunk_size = 25, 4
        criterion.mask_centerline_ce_radius = 3
        data = RFDETRDataModule(mc, tc)
        data.setup("fit")
        dataset = data._dataset_train
        assert isinstance(dataset, CocoDetection)
        if len(cohort) == 16:
            cohort += random_training_cohort(dataset, cohort)
            write_json(
                output / "protocol.json",
                dict(
                    git_commit=expected_commit,
                    checkpoints=checkpoints,
                    cohort=cohort,
                    coefficient_probe=0.1,
                    gt_iterations=25,
                    chunk_size=4,
                    radius=3,
                    eta=1,
                    modes=["gt_centerline", *CE_MODES],
                    mask_seeds=[420, 421, 422],
                    random_seed=20261117,
                    batch_size=8,
                    scale=504,
                    groups=13,
                    optimizer_steps=0,
                    selection="random training GTs, without prediction/validation filtering",
                    limits="local shared-feature derivatives; excludes upstream, spatial, box and auxiliary paths",
                ),
            )
        for batch in range(4):
            probe_batch(
                module,
                dataset,
                cohort[batch * 8 : batch * 8 + 8],
                batch,
                target,
                gt_centerline=True,
                split_mask=True,
                centerline_ce=True,
            )
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
        del module, data
        gc.collect()
        torch.cuda.empty_cache()
    summarize(output, cohort)
    write_json(
        output / "completed.json",
        dict(complete=True, checkpoint_hashes_unchanged=True, optimizer_steps=0, parameter_gradients=False),
    )
    print("CE_FROZEN_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.previous, args.output, args.expected_commit)
