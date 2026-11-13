# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Protect the matched ablation and its unchanged scheduler horizon."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pytest_mock import MockerFixture

from scripts.tomato_short_ablation import REPO, StopAfterTen, short_config


def test_short_configs_only_change_topology_weight(tmp_path: Path) -> None:
    """The two arms keep identical training settings and preserve the full schedule."""
    base = yaml.safe_load((REPO / "configs/tomato/seg_small_baseline.yaml").read_text())
    original = copy.deepcopy(base)
    baseline = short_config(base, tmp_path, 0.0)
    topology = short_config(base, tmp_path, 0.1)
    assert base == original
    baseline["model"]["train_config"]["init_args"]["mask_cldice_loss_coef"] = 0.1
    assert baseline == topology
    effective = topology["model"]["train_config"]["init_args"]
    for key, value in original["model"]["train_config"]["init_args"].items():
        if key != "output_dir":
            assert effective[key] == value
    assert "max_epochs" not in topology["trainer"]


@pytest.mark.parametrize("epoch", [0, 8, 9])
def test_stop_only_after_tenth_epoch(epoch: int, tmp_path: Path) -> None:
    """Stopping must not shorten the horizon used to derive optimizer schedules."""
    trainer = SimpleNamespace(current_epoch=epoch, max_epochs=100, should_stop=False)
    callback = StopAfterTen(str(tmp_path))
    callback.on_train_epoch_end(trainer, None)
    assert trainer.should_stop == (epoch == 9)
    assert trainer.max_epochs == 100


def test_evaluate_action_never_starts_training(tmp_path: Path, mocker: MockerFixture) -> None:
    """An interrupted evaluation can be completed without repeating either training arm."""
    from scripts import tomato_short_ablation

    mocker.patch("sys.argv", ["runner", "evaluate", "--output", str(tmp_path), "--expected-commit", "verified"])
    train = mocker.patch.object(tomato_short_ablation, "run")
    evaluate = mocker.patch.object(tomato_short_ablation, "evaluate_pair")
    tomato_short_ablation.main()
    train.assert_not_called()
    evaluate.assert_called_once_with(tmp_path, "verified")


def test_recall_configs_only_change_beta(tmp_path: Path) -> None:
    """Recall arms retain the same weight, schedule, batch and model configuration."""
    base = yaml.safe_load((REPO / "configs/tomato/seg_small_baseline.yaml").read_text())
    original = copy.deepcopy(base)
    symmetric = short_config(base, tmp_path, 0.1, recall_beta=1.0)
    recall = short_config(base, tmp_path, 0.1, recall_beta=2.0)
    symmetric["model"]["train_config"]["init_args"]["mask_cldice_recall_beta"] = 2.0
    assert symmetric == recall
    assert base == original


def test_recall_pair_cli_dispatch(tmp_path: Path, mocker: MockerFixture) -> None:
    """Explicit recall comparison reaches the runner; probe-only beta cannot silently select training."""
    from scripts import tomato_short_ablation

    mocker.patch(
        "sys.argv",
        ["runner", "run", "--output", str(tmp_path), "--expected-commit", "verified", "--comparison", "recall-beta"],
    )
    train = mocker.patch.object(tomato_short_ablation, "run")
    tomato_short_ablation.main()
    train.assert_called_once_with(tmp_path, "verified", "recall-beta")
