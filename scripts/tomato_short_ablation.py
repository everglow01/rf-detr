# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run the authorized K25 gradient probe and matched ten-epoch clDice ablation.

Keep the 100-epoch scheduler horizon and stop at epoch ten with a callback. Reuse the committed timing and validation
helpers from the original experiment. New outputs are exclusive; existing E0/E1 artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from pytorch_lightning import Callback

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from scripts.diagnose_cldice_dedup import batch_at, cpu_grads, parameter_hash
from scripts.diagnose_tomato_e1 import configurations, gradient_relation, group_name, write_json

REPO = Path(__file__).resolve().parents[1]
TIMING_DIR = REPO / "reports/tomato-20261111/runs/e1-seg-small-cldice-c0p1-v1-seed42"
EVALUATOR = REPO / "reports/tomato-20261111/baseline-analysis/evaluate_stems.py"
E1 = REPO / "output/output1/e1-seg-small-cldice-c0p1-v1-seed42"
ARM_CONFIGS = {
    "topology": (("baseline", 0.0, 1.0), ("cldice", 0.1, 1.0)),
    "recall-beta": (("beta1", 0.1, 1.0), ("beta2", 0.1, 2.0)),
}


class StopAfterTen(Callback):
    """Bound training without shortening scheduler horizons; record initialization and batch parity."""

    def __init__(self, output_dir: str) -> None:
        """Write one independent audit per distributed rank."""
        self.output = Path(output_dir)
        self.audit: dict[str, Any] = {"epochs": []}
        self.training_memory: list[dict[str, Any]] = []

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Require the original schedule and fingerprint parameters before updates."""
        assert trainer.max_epochs == pl_module.train_config.epochs == 100
        assert trainer.world_size == 2
        self.audit["initial_parameter_sha256"] = parameter_hash(pl_module.model)
        self.audit["schedule_epochs"] = trainer.max_epochs

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Never permit an eleventh training epoch."""
        assert trainer.current_epoch < 10

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        """Record each epoch's first transformed batch and learning rates without consuming RNG."""
        if batch_idx:
            return
        samples, targets = batch
        digest = hashlib.sha256(samples.tensors.detach().cpu().contiguous().numpy().tobytes())
        for target in targets:
            for key in ("image_id", "labels", "boxes", "masks"):
                digest.update(target[key].detach().cpu().contiguous().numpy().tobytes())
        self.audit["epochs"].append(
            dict(
                epoch=trainer.current_epoch + 1,
                first_batch_sha256=digest.hexdigest(),
                learning_rates=[group["lr"] for group in trainer.optimizers[0].param_groups],
                step=trainer.global_step,
            )
        )
        write_json(self.output / f"audit.rank{trainer.global_rank}.json", self.audit)

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        """Separate training peaks from validation peaks for a fair memory comparison."""
        if trainer.sanity_checking:
            return
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.training_memory.append(
            dict(
                epoch=trainer.current_epoch + 1,
                peak_allocated_mib=torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2**20,
            )
        )
        write_json(self.output / f"training-memory.rank{trainer.global_rank}.json", self.training_memory)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Request the same epoch-boundary stop on both ranks after complete validation."""
        if trainer.current_epoch == 9:
            trainer.should_stop = True

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Confirm ten epochs and retain final update counts."""
        assert len(self.audit["epochs"]) == 10
        self.audit.update(complete=True, final_global_step=trainer.global_step)
        write_json(self.output / f"audit.rank{trainer.global_rank}.json", self.audit)


def short_config(base: dict[str, Any], output: Path, coefficient: float, recall_beta: float = 1.0) -> dict[str, Any]:
    """Set the explicit ablation values while keeping the 100-epoch plan intact."""
    config = copy.deepcopy(base)
    train = config["model"]["train_config"]["init_args"]
    assert train["epochs"] == 100 and train["devices"] == 2 and train["batch_size"] == 8
    train.update(
        output_dir=str(output),
        mask_cldice_loss_coef=coefficient,
        mask_cldice_class_ids=[1],
        mask_cldice_iterations=25,
        mask_cldice_chunk_size=4,
        mask_cldice_recall_beta=recall_beta,
    )
    config["trainer"] = dict(
        enable_model_summary=False,
        callbacks=[
            dict(class_path="scripts.tomato_short_ablation.StopAfterTen", init_args=dict(output_dir=str(output))),
            dict(class_path="benchmark_callbacks.BenchmarkTiming", init_args=dict(output_dir=str(output))),
        ],
    )
    return config


def probe(output: Path, recall_beta: float = 1.0) -> None:
    """Complete the old OOM batch's native/region/topology shared-gradient diagnosis without updates."""
    output.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(
        config=E1 / "training.yaml", checkpoint=E1 / "training/checkpoint_best_ema.pth", output=output
    )
    mc, tc = configurations(args)
    tc.mask_cldice_iterations, tc.mask_cldice_chunk_size = 25, 4
    tc.mask_cldice_recall_beta = recall_beta
    assert tc.mask_cldice_loss_coef == 0.1 and tc.mask_cldice_class_ids == [1]
    torch.set_num_threads(2)
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
    before = parameter_hash(module.model)
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    assert isinstance(data._dataset_train, CocoDetection)
    samples, targets, meta = batch_at(data._dataset_train, 2, mc.patch_size * mc.num_windows)
    assert meta["dataset_indices"] == [558, 334, 1376, 278, 933, 839, 1035, 8]
    named = [(name, p) for name, p in module.model.named_parameters() if p.requires_grad]
    params: list[torch.Tensor] = [p for _, p in named]
    torch.manual_seed(7002)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module.model(samples, targets)
        outputs["mask_padding"] = meta["padding"]
        losses = module.criterion(outputs, targets)
        weighted = {
            key: value * module.criterion.weight_dict[key]
            for key, value in losses.items()
            if key in module.criterion.weight_dict
        }
        native = torch.stack([value for key, value in weighted.items() if key != "loss_mask_cldice"]).sum()
        region = torch.stack(
            [value for key, value in weighted.items() if key.startswith(("loss_mask_ce", "loss_mask_dice"))]
        ).sum()
        topology = weighted["loss_mask_cldice"]
    native_grad = cpu_grads(native, params, retain_graph=True)
    region_grad = cpu_grads(region, params, retain_graph=True)
    topo_grad = cpu_grads(topology, params, retain_graph=False)
    for values in (native_grad, region_grad, topo_grad):
        assert all(value is None or torch.isfinite(value).all() for value in values)
    groups = {group_name(name) for name, _ in named} | {"all"}
    stats = {}
    for group in sorted(groups):
        ids = [i for i, (name, _) in enumerate(named) if group == "all" or group_name(name) == group]
        stats[group] = dict(
            original_vs_topology=gradient_relation([native_grad[i] for i in ids], [topo_grad[i] for i in ids]),
            region_vs_topology=gradient_relation([region_grad[i] for i in ids], [topo_grad[i] for i in ids]),
        )
    assert all(
        stats[group]["original_vs_topology"]["second_norm"] > 0 for group in ("backbone", "transformer", "mask_head")
    )
    assert before == parameter_hash(module.model)
    result = dict(
        passed=True,
        batch=2,
        dataset_indices=meta["dataset_indices"],
        scale=504,
        iterations=25,
        coefficient=0.1,
        chunk_size=4,
        recall_beta=recall_beta,
        batch_size=8,
        group_detr=13,
        native_loss=float(native.detach()),
        region_loss=float(region.detach()),
        topology_loss=float(topology.detach()),
        groups=stats,
        optimizer_steps=0,
        parameters_unchanged=True,
        parameter_sha256=before,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    write_json(output / "completed.json", result)
    print(json.dumps(result), flush=True)


def train_arm(config: Path, output: Path, env: dict[str, str]) -> None:
    """Launch the unchanged official DDP CLI and retain wall time plus GPU telemetry."""
    output.mkdir(exist_ok=False)
    command = [
        sys.executable,
        "-B",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--module",
        "rfdetr",
        "fit",
        "--config",
        str(config),
    ]
    start = time.perf_counter()
    write_json(
        output / "process.json", dict(command=command, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    )
    with (output / "train.log").open("w") as log, (output / "gpu-samples.csv").open("w") as telemetry:
        telemetry.write("elapsed_seconds,index,memory_used_mib,utilization_gpu_percent,power_watts\n")
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            sample = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            for line in sample.splitlines():
                telemetry.write(f"{time.perf_counter() - start:.3f},{line}\n")
            telemetry.flush()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    result = dict(
        command=command,
        exit_code=process.returncode,
        wall_seconds=time.perf_counter() - start,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    write_json(output / "process.json", result)
    if process.returncode:
        raise RuntimeError(f"Training failed: {output / 'train.log'}")
    print(json.dumps(dict(stage="training_complete", arm=output.name, **result)), flush=True)


def load_evaluator() -> Any:
    """Reuse the existing isolated evaluation dependencies and fail before starting training if unavailable."""
    dependencies = REPO / "output/output1/baseline-analysis-20261111/eval-deps"
    if dependencies.is_dir() and str(dependencies) not in sys.path:
        sys.path.insert(0, str(dependencies))
    return runpy.run_path(str(EVALUATOR))["run"]


def run(output: Path, expected_commit: str, comparison: str = "topology") -> None:
    """Gate the paired short runs on finite connected gradients, then evaluate both last and best EMA weights."""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    load_evaluator()
    arms = ARM_CONFIGS[comparison]
    output.mkdir(parents=True, exist_ok=False)
    base_path = REPO / "configs/tomato/seg_small_baseline.yaml"
    base = yaml.safe_load(base_path.read_text())
    paths = [
        REPO / base["model"]["model_config"]["init_args"]["pretrain_weights"],
        base_path,
        Path(__file__),
        TIMING_DIR / "benchmark_callbacks.py",
        EVALUATOR,
    ]
    paths.extend(
        REPO / "data/0/TOMATO.v7i.rfdetr-keypoints-v1" / split / "_annotations.coco.json"
        for split in ("train", "valid")
    )
    write_json(
        output / "provenance.json",
        dict(
            git_commit=expected_commit,
            comparison=comparison,
            arms=arms,
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            planned_epochs=10,
            scheduler_horizon=100,
            world_size=2,
            per_gpu_batch=8,
            effective_batch=16,
            primary_comparison="last EMA at epoch 10",
            secondary_comparison="best EMA within epochs 1-10",
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
        PYTHONPATH=os.pathsep.join((str(REPO / "src"), str(TIMING_DIR))),
    )
    with (output / "probe.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "scripts.tomato_short_ablation",
                "probe",
                "--output",
                str(output / "probe"),
                "--recall-beta",
                str(max(beta for _, _, beta in arms)),
            ],
            cwd=REPO,
            env=dict(env, CUDA_VISIBLE_DEVICES="0"),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert json.loads((output / "probe/completed.json").read_text())["passed"]
    print("GRADIENT_PROBE_PASSED", flush=True)
    for name, coefficient, recall_beta in arms:
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
        config = output / f"{name}.yaml"
        config.write_text(yaml.safe_dump(short_config(base, output / name, coefficient, recall_beta), sort_keys=False))
        train_arm(config, output / name, env)
    evaluate_pair(output, expected_commit)


def evaluate_pair(output: Path, expected_commit: str) -> None:
    """Evaluate an already completed pair without creating an optimizer or starting another training run."""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit
    evaluate = load_evaluator()
    provenance = json.loads((output / "provenance.json").read_text())
    arms = [name for name, _, _ in ARM_CONFIGS[provenance.get("comparison", "topology")]]
    for rank in (0, 1):
        audits = [json.loads((output / name / f"audit.rank{rank}.json").read_text()) for name in arms]
        assert audits[0] == audits[1], "Initialization, sampled first batches, learning rates or update counts differ"
        assert audits[0]["complete"] and len(audits[0]["epochs"]) == 10
    write_json(
        output / "paired_training_passed.json",
        dict(passed=True, epochs=10, initialization_and_first_batches_match=True),
    )
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    assert not any(evaluation.iterdir()), "Keep existing evaluation results; choose a fresh analysis before retrying"
    before_weights = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in arms
        for p in (output / name).glob("*.pth")
    }
    write_json(
        output / "evaluation-provenance.json",
        dict(
            training_git_commit=json.loads((output / "provenance.json").read_text())["git_commit"],
            evaluation_git_commit=expected_commit,
            evaluator_sha256=hashlib.sha256(EVALUATOR.read_bytes()).hexdigest(),
            skimage_version=getattr(sys.modules["skimage"], "__version__"),
            skimage_path=getattr(sys.modules["skimage"], "__file__"),
            checkpoint_sha256=before_weights,
            optimizer_steps=0,
        ),
    )
    torch.set_num_threads(2)
    for name in arms:
        for file, label in (("last_ema.pth", "last10"), ("checkpoint_best_ema.pth", "best")):
            torch.manual_seed(42)
            evaluate(output / name / file, f"{name}-{label}", evaluation, 0, "cuda")
            torch.cuda.empty_cache()
    weights = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in arms
        for p in (output / name).glob("*.pth")
    }
    assert weights == before_weights, "Evaluation changed a checkpoint"
    write_json(output / "completed.json", dict(complete=True, epochs_per_arm=10, evaluations=4, weight_sha256=weights))


def main() -> None:
    """Run one isolated probe or the complete user-authorized experiment pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "run", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--recall-beta", type=float, default=1.0, help="F-beta balance for the frozen probe only")
    parser.add_argument(
        "--comparison",
        choices=tuple(ARM_CONFIGS),
        default="topology",
        help="Training pair: topology on/off or beta=1 versus beta=2",
    )
    args = parser.parse_args()
    if args.comparison != "topology" and args.action != "run":
        parser.error("--comparison is for run; evaluate reads the archived provenance")
    if args.recall_beta != 1.0 and args.action != "probe":
        parser.error("--recall-beta is only supported for the frozen probe")
    if args.action == "probe":
        probe(args.output, args.recall_beta)
    elif args.action == "evaluate":
        evaluate_pair(args.output, args.expected_commit)
    else:
        run(args.output, args.expected_commit, args.comparison)


if __name__ == "__main__":
    main()
