# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run the authorized baseline/centerline-CE/local-CE ten-epoch ablation with a frozen coefficient."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from scripts.diagnose_cldice_dedup import batch_at, parameter_hash
from scripts.diagnose_tomato_e1 import configurations, group_name, write_json
from scripts.diagnose_tomato_recall import evaluate as evaluate_recall
from scripts.tomato_short_ablation import EVALUATOR, REPO, TIMING_DIR, load_evaluator, short_config, train_arm

ARMS = ("baseline", "centerline_ce", "centerline_local_ce")


def ce_config(base: dict[str, Any], output: Path, arm: str, coefficient: float) -> dict[str, Any]:
    """Change only the optional final stem loss; B and C use an identical calibrated outer coefficient."""
    assert arm in ARMS
    config = short_config(base, output, 0 if arm == "baseline" else coefficient)
    train = config["model"]["train_config"]["init_args"]
    train.update(mask_cldice_mode="cldice" if arm == "baseline" else arm, mask_centerline_ce_radius=3)
    return config


def backward_smoke(config: Path, output: Path) -> None:
    """Check finite full-model backward on the historical large batch, without creating an optimizer."""
    mc, tc = configurations(argparse.Namespace(config=config, checkpoint=None, output=output))
    tc.amp_dtype = None
    torch.manual_seed(42)
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
    before = parameter_hash(module.model)
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    assert isinstance(data._dataset_train, CocoDetection)
    samples, targets, meta = batch_at(data._dataset_train, 2, mc.patch_size * mc.num_windows)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module.model(samples, targets)
        outputs["mask_padding"] = meta["padding"]
        # Backpropagate the new term alone to prove it reaches all shared trainable model parts.
        indices = module.criterion.matcher(outputs, targets, group_detr=module.criterion.group_detr)
        key = module.criterion.mask_topology_loss_key
        loss = module.criterion.loss_mask_cldice(outputs, targets, indices)[key] * module.criterion.weight_dict[key]
    torch.autograd.backward(loss)
    norms: dict[str, float] = {}
    for name, parameter in module.model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            group = group_name(name)
            norms[group] = norms.get(group, 0.0) + float(parameter.grad.float().square().sum())
    assert all(norms.get(name, 0) > 0 for name in ("backbone", "transformer", "mask_head"))
    assert before == parameter_hash(module.model)
    write_json(
        output / "backward-smoke.json",
        dict(
            passed=True,
            optimizer_steps=0,
            parameters_unchanged=True,
            loss=float(loss.detach()),
            squared_gradient_norms=norms,
            dataset_indices=meta["dataset_indices"],
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        ),
    )
    del module, data, samples, targets, outputs, loss
    gc.collect()
    torch.cuda.empty_cache()
    print("CE_FULL_BACKWARD_SMOKE_PASSED", flush=True)


def evaluate_all(output: Path, expected_commit: str) -> None:
    """Evaluate fixed epoch-ten EMA at native thresholds and a common baseline precision floor."""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
    for rank in (0, 1):
        audits = [json.loads((output / arm / f"audit.rank{rank}.json").read_text()) for arm in ARMS]
        assert all(a == audits[0] for a in audits), (
            "Initialization, first batches, learning rates or update counts differ"
        )
        assert audits[0]["complete"] and len(audits[0]["epochs"]) == 10
    write_json(output / "paired_training_passed.json", dict(passed=True, epochs=10, arms=ARMS))
    checkpoints = {arm: output / arm / "last_ema.pth" for arm in ARMS}
    before = {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in checkpoints.items()}
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=False)
    recall = output / "recall"
    recall.mkdir(exist_ok=False)
    evaluate = load_evaluator()
    torch.set_num_threads(2)
    summaries = {}
    for arm in ARMS:
        torch.manual_seed(42)
        evaluate(checkpoints[arm], f"{arm}-last10", evaluation, 0, "cuda")
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        summaries[arm] = evaluate_recall(
            output / f"{arm}.yaml", checkpoints[arm], recall / arm, evaluation / f"{arm}-last10"
        )
    floor = summaries["baseline"]["fixed_0p3"]["precision"]
    comparable = {}
    for arm in ARMS:
        curve = json.loads((recall / arm / "mask_pr_curve.json").read_text())
        choices = [point for point in curve if point["precision"] >= floor]
        comparable[arm] = max(choices, key=lambda p: (p["recall"], p["threshold"])) if choices else None
    assert before == {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in checkpoints.items()}
    write_json(
        output / "completed.json",
        dict(
            complete=True,
            epochs_per_arm=10,
            checkpoint_sha256=before,
            target_precision=floor,
            max_recall_at_precision_floor=comparable,
            summaries=summaries,
            evaluation_optimizer_steps=0,
            checkpoint_hashes_unchanged=True,
        ),
    )
    print("CE_SHORT_EXPERIMENT_COMPLETE " + json.dumps(comparable), flush=True)


def run(output: Path, frozen: Path, expected_commit: str) -> None:
    """Train three serial matched runs only after the completed frozen probe and finite full backward."""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert json.loads((frozen / "completed.json").read_text())["complete"]
    assert json.loads((frozen / "protocol.json").read_text())["git_commit"] == expected_commit
    calibration = json.loads((frozen / "calibration.json").read_text())
    coefficient = calibration["coefficient"]
    assert calibration["finite"] and 0 < coefficient < float("inf")
    load_evaluator()
    output.mkdir(parents=True, exist_ok=False)
    base_path = REPO / "configs/tomato/seg_small_baseline.yaml"
    base = yaml.safe_load(base_path.read_text())
    assert not base["model"]["model_config"]["init_args"].get("query_point_head", False)
    paths = [
        base_path,
        EVALUATOR,
        TIMING_DIR / "benchmark_callbacks.py",
        frozen / "calibration.json",
        REPO / base["model"]["model_config"]["init_args"]["pretrain_weights"],
    ]
    paths += [
        REPO / "data/0/TOMATO.v7i.rfdetr-keypoints-v1" / split / "_annotations.coco.json"
        for split in ("train", "valid")
    ]
    write_json(
        output / "provenance.json",
        dict(
            git_commit=expected_commit,
            arms=ARMS,
            coefficient=coefficient,
            frozen=str(frozen),
            planned_epochs=10,
            scheduler_horizon=100,
            primary_comparison="fixed epoch 10 EMA",
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        ),
    )
    subprocess.run(
        [
            "git",
            "archive",
            "-o",
            str(output / "source.tar.gz"),
            "HEAD",
            "src",
            "scripts",
            "configs/tomato",
            str(TIMING_DIR.relative_to(REPO)),
            str(EVALUATOR.relative_to(REPO)),
        ],
        check=True,
    )
    for arm in ARMS:
        (output / f"{arm}.yaml").write_text(
            yaml.safe_dump(ce_config(base, output / arm, arm, coefficient), sort_keys=False)
        )
    torch.set_num_threads(2)
    backward_smoke(output / "centerline_local_ce.yaml", output)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        RFDETR_SKIP_DINOV2_PREWARM="1",
        HF_HUB_OFFLINE="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join((str(REPO / "src"), str(REPO), str(TIMING_DIR))),
    )
    for arm in ARMS:
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
        print(f"CE_TRAIN_START arm={arm} epochs=10 coefficient={0 if arm == 'baseline' else coefficient}", flush=True)
        train_arm(output / f"{arm}.yaml", output / arm, env, stream_log=True)
    evaluate_all(output, expected_commit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.output, args.frozen, args.expected_commit)
