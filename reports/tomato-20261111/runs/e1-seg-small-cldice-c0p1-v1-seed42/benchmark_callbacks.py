"""Observe the unchanged RF-DETR Lightning CLI at epoch boundaries."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback


class BenchmarkTiming(Callback):
    """Record per-rank training, validation, complete-epoch time and CUDA peaks."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.fit_started = 0.0

    def _sync(self, trainer: Any) -> float:
        """Complete this rank's queued CUDA work before a boundary timestamp."""
        torch.cuda.synchronize(trainer.strategy.root_device)
        return time.perf_counter()

    def _write(self, trainer: Any, status: str) -> None:
        """Write an atomic snapshot independently for each rank."""
        payload = {
            "status": status, "rank": trainer.global_rank, "world_size": trainer.world_size,
            "precision": trainer.precision, "epoch_records": self.records,
            "fit_elapsed_seconds": time.perf_counter() - self.fit_started,
            "note": "epoch_total_seconds spans epoch start to next epoch start (last: fit end), including validation/checkpoints",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"timing.rank{trainer.global_rank}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(target)

    def _finish_epoch(self, trainer: Any, now: float) -> None:
        """Finalize the previous epoch after all its checkpoint hooks have finished."""
        if self.current is None:
            return
        self.current["epoch_total_seconds"] = now - self.current["started_perf_counter"]
        self.current["peak_allocated_mib"] = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2**20
        self.current["peak_reserved_mib"] = torch.cuda.max_memory_reserved(trainer.strategy.root_device) / 2**20
        self.current["complete"] = True
        self._write(trainer, "running")
        if trainer.is_global_zero:
            print("BENCH_EPOCH " + json.dumps(self.current), flush=True)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Start fit timing after initialization; the launcher separately times full process wall time."""
        self.fit_started = self._sync(trainer)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Finalize the last epoch and begin the next without per-step synchronization."""
        now = self._sync(trainer)
        self._finish_epoch(trainer, now)
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
        self.current = {
            "epoch": trainer.current_epoch + 1, "started_perf_counter": now,
            "global_step_start": trainer.global_step, "expected_train_batches": trainer.num_training_batches,
            "train_batches": 0, "complete": False,
        }
        self.records.append(self.current)
        self._write(trainer, "running")

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        """Count batches and publish occasional progress without synchronizing CUDA."""
        self.current["train_batches"] = batch_idx + 1
        if trainer.is_global_zero and (batch_idx + 1) % 50 == 0:
            print(f"BENCH_PROGRESS epoch={trainer.current_epoch + 1} batch={batch_idx + 1} "
                  f"of={trainer.num_training_batches} step={trainer.global_step}", flush=True)

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        """Measure training time up to the beginning of full validation."""
        if trainer.sanity_checking or self.current is None:
            return
        now = self._sync(trainer)
        self.current["validation_started_perf_counter"] = now
        self.current["training_seconds"] = now - self.current["started_perf_counter"]
        self.current["optimizer_steps"] = trainer.global_step - self.current["global_step_start"]
        self._write(trainer, "validating")
        if trainer.is_global_zero:
            print(f"BENCH_TRAIN_DONE epoch={trainer.current_epoch + 1} "
                  f"seconds={self.current['training_seconds']:.3f}", flush=True)

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        """Include COCO metric computation; checkpoint overhead remains in complete-epoch timing."""
        if trainer.sanity_checking or self.current is None:
            return
        now = self._sync(trainer)
        self.current["validation_seconds"] = now - self.current["validation_started_perf_counter"]
        names = ("val/segm_mAP_50_95", "val/ema_segm_mAP_50_95", "val/mAP_50_95", "val/loss")
        self.current["metrics"] = {
            name: float(trainer.callback_metrics[name]) for name in names if name in trainer.callback_metrics
        }
        self._write(trainer, "epoch_finishing")

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Finish the final epoch and save final rank-specific timing evidence."""
        self._finish_epoch(trainer, self._sync(trainer))
        self._write(trainer, "completed")

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        """Preserve incomplete timing if the original training loop fails."""
        if self.fit_started:
            self._write(trainer, "failed")
