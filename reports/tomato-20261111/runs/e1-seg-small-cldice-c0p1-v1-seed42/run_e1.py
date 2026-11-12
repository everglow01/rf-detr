"""Run E1 from a hashed snapshot of the current experiment source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    """Keep exact command/config provenance, process wall time, and sampled device telemetry."""
    root = Path(__file__).resolve().parent
    project = Path("/usr/local/rfdetr26")
    python = project / ".venv/bin/python"
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True, help="Full SHA of the snapshot base commit")
    parser.add_argument("--expected-source-sha256", required=True, help="SHA256 of source_manifest.json")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    if len(args.expected_commit) != 40 or commit != args.expected_commit:
        raise RuntimeError("Remote HEAD must match the recorded snapshot base commit")
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=project, text=True).strip()
    manifest_path = root / "source_manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != args.expected_source_sha256:
        raise RuntimeError("Source manifest differs from the pinned experiment snapshot")
    source_manifest = json.loads(manifest_path.read_text())
    assert source_manifest["base_commit"] == commit
    for relative, expected in source_manifest["files"].items():
        if hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Snapshot file changed: {relative}")
    results = []
    for label, devices, visible in (("training", 2, "0,1"),):
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
        ).strip()
        if processes:
            raise RuntimeError(f"GPU compute processes present before {label}: {processes}; no training started")
        out = root / label
        out.mkdir(exist_ok=False)
        config = root / f"{label}.yaml"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", RFDETR_SKIP_DINOV2_PREWARM="1")
        env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "source/src")])
        command = [str(python), "-B", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc_per_node={devices}", "--module", "rfdetr", "fit", "--config", str(config)]
        started = time.perf_counter()
        metadata = {"git_base_commit": commit, "source_state": "uncommitted_snapshot",
                    "source_manifest_sha256": args.expected_source_sha256,
                    "label": label, "devices": devices, "cuda_visible_devices": visible,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
        (out / "process.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("BENCH_RUN_START " + json.dumps(metadata), flush=True)
        with (out / "train.log").open("w") as log, (out / "gpu-samples.csv").open("w") as telemetry:
            telemetry.write("elapsed_seconds,index,name,memory_used_mib,utilization_gpu_percent,power_watts\n")
            process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                sample = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                )
                if sample.returncode == 0:
                    for line in sample.stdout.splitlines():
                        telemetry.write(f"{time.perf_counter() - started:.3f},{line}\n")
                    telemetry.flush()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        metadata.update(exit_code=process.returncode, wall_seconds=time.perf_counter() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        (out / "process.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results.append(metadata)
        (root / "process-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("BENCH_RUN_END " + json.dumps(metadata), flush=True)
        if process.returncode:
            print("\n".join((out / "train.log").read_text(errors="replace").splitlines()[-65:]), flush=True)
            sys.exit(process.returncode)
    print("E1_100_EPOCHS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
