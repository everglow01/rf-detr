# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Real two-process reduction and device-safe Hungarian indices for point loss."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tests._markers import _requires_cuda, _requires_gloo_spawn
from tests.models.test_query_points import _criterion


def _worker(rank: int, rendezvous: str, directory: str, labels: tuple[int, int]) -> None:
    """Compute one matched point per rank, selected by class.

    Examples:
        Requires two processes:
        >>> _worker(0, 'file:///tmp/init', '/tmp', (1, 0))  # doctest: +SKIP
    """
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(nn.Linear(1, 2, bias=False))
        nn.init.constant_(model.module.weight, 0.5)
        pred = model(torch.ones(1, 1)).unsqueeze(0)
        target = {"labels": torch.tensor([labels[rank]]), "keypoints": torch.tensor([[[0.1, 0.3, 2.0]]])}
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = _criterion().loss_query_point({"pred_pick_points": pred}, [target], indices)["loss_query_point"]
        loss.backward()
        torch.save({"loss": loss.detach(), "grad": model.module.weight.grad}, Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestQueryPointDistributed:
    """DDP averaging must agree with a single global valid-point mean."""

    @pytest.mark.ddp
    @_requires_gloo_spawn
    @pytest.mark.parametrize("labels", [(1, 1), (1, 0), (0, 0)])
    def test_two_rank_matches_serial(self, tmp_path: Path, labels: tuple[int, int]) -> None:
        torch.multiprocessing.spawn(_worker, args=((tmp_path / "init").as_uri(), str(tmp_path), labels), nprocs=2)
        rows = [torch.load(tmp_path / f"rank{i}.pt", weights_only=True) for i in range(2)]
        expected = 0.3 if 1 in labels else 0.0
        torch.testing.assert_close(sum(r["loss"] for r in rows) / 2, torch.tensor(expected))
        for row in rows:
            torch.testing.assert_close(row["grad"], torch.full((2, 1), 0.5 if 1 in labels else 0.0))

    @pytest.mark.gpu
    @_requires_cuda
    def test_cpu_match_indices_with_cuda_targets(self) -> None:
        pred = torch.full((1, 2, 2), 0.5, device="cuda", requires_grad=True)
        target = {
            "labels": torch.tensor([0, 1], device="cuda"),
            "keypoints": torch.tensor([[[0.0, 0.0, 0.0]], [[0.1, 0.3, 2.0]]], device="cuda"),
        }
        indices = [(torch.arange(2), torch.arange(2))]
        loss = _criterion().loss_query_point({"pred_pick_points": pred}, [target], indices)["loss_query_point"]
        loss.backward()
        assert torch.isfinite(pred.grad).all() and pred.grad[0, 1].abs().sum() > 0
