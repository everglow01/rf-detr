# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify GT deduplication before bounded K=25 probes; never create an optimizer.

Run as ``python -m scripts.diagnose_cldice_dedup`` from the repository root. The legacy criterion is loaded from the
immutable E1 archive, while all model and dataset code comes from the checked-out Git revision. K=25 stages run only
after exact K=10 loss/mask-input-gradient parity succeeds on every selected batch. Shared-network backward repeatability
is measured separately because CUDA grid_sample does not support deterministic backward.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.datasets.coco import CocoDetection
from rfdetr.models import criterion as criterion_module
from rfdetr.models.soft_cldice import _soft_skeleton
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import make_collate_fn
from scripts.diagnose_tomato_e1 import (
    configurations,
    gradient_relation,
    group_name,
    seed_sample,
    skeleton_trace,
    write_json,
)


def legacy_criterion(path: Path, current: Any) -> Any:
    """Bind only the archived clDice adapter; preserve native losses and matcher."""
    spec = importlib.util.spec_from_file_location("tomato_archived_criterion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = copy.copy(current)
    result._cldice_on_image = types.MethodType(module.SetCriterion._cldice_on_image, result)
    result.loss_mask_cldice = types.MethodType(module.SetCriterion.loss_mask_cldice, result)
    return result


def batch_at(dataset: CocoDetection, index: int, block: int) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Replay the previous audit's augmented samples and balanced input scales."""
    order = np.random.default_rng(20261112).permutation(len(dataset)).tolist()
    ids = order[index * 8 : (index + 1) * 8]
    examples = []
    for item in ids:
        seed_sample(42, item)
        examples.append(dataset[item])
    samples, targets = make_collate_fn(block_size=block, pack=False)(examples)
    samples = samples.to("cuda")
    targets = [{k: v.cuda() if isinstance(v, Tensor) else v for k, v in t.items()} for t in targets]
    padding = samples.mask
    scale = (264, 384, 504)[index % 3]
    samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
    samples.mask = F.interpolate(samples.mask[:, None].float(), size=scale, mode="nearest")[:, 0].bool()
    return samples, targets, dict(padding=padding, scale=scale, dataset_indices=ids)


def cpu_grads(loss: Tensor, params: list[Tensor], *, retain_graph: bool) -> list[Tensor | None]:
    """Take one derivative, returning detached CPU copies to bound GPU memory."""
    values = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return [value.detach().float().cpu() if value is not None else None for value in values]


def gradient_difference(first: list[Tensor | None], second: list[Tensor | None]) -> dict[str, Any]:
    """Check every shared parameter, not just the norm of the full vector."""
    error, scale, maximum, exact = 0.0, 0.0, 0.0, True
    for left, right in zip(first, second, strict=True):
        assert (left is None) == (right is None), "Gradient connectivity changed"
        if left is None or right is None:
            continue
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        difference = left - right
        maximum = max(maximum, float(difference.abs().max()))
        error += float(difference.square().sum())
        scale += float(left.square().sum())
        exact &= torch.equal(left, right)
    return dict(
        max_absolute_error=maximum,
        relative_l2_error=math.sqrt(error / max(scale, 1e-30)),
        bitwise_equal=exact,
        **gradient_relation(first, second),
    )


def verify_gradient_parity(
    first: list[Tensor | None],
    first_repeat: list[Tensor | None],
    second: list[Tensor | None],
    second_repeat: list[Tensor | None],
) -> dict[str, Any]:
    """Bound cross-path error by repeated-backward controls and a strict global cap.

    BF16 shared-backbone backward can vary between identical calls. Require each parameter's maximum error to stay
    within four times its observed self-repeat noise (absolute floor 1e-7), AND global relative L2 error <=5e-5. A noisy
    control never waives the global cap. Scalar losses retain the original tight check.
    """
    cross = gradient_difference(first, second)
    control_first = gradient_difference(first, first_repeat)
    control_second = gradient_difference(second, second_repeat)
    failures = []
    for index, (left, left_repeat, right, right_repeat) in enumerate(
        zip(first, first_repeat, second, second_repeat, strict=True)
    ):
        if left is None:
            continue
        assert left_repeat is not None and right is not None and right_repeat is not None
        noise = max(float((left - left_repeat).abs().max()), float((right - right_repeat).abs().max()))
        error = float((left - right).abs().max())
        allowed = max(1e-7, 4 * noise)
        if error > allowed:
            failures.append(dict(parameter_index=index, maximum_error=error, observed_noise=noise, allowed=allowed))
    return dict(
        passed=not failures and cross["relative_l2_error"] <= 5e-5,
        relative_l2_limit=5e-5,
        noise_multiplier=4,
        absolute_floor=1e-7,
        cross=cross,
        current_repeat=control_first,
        alternate_repeat=control_second,
        out_of_bound_parameters=failures,
    )


@torch.no_grad()
def prediction_skeleton_mass(
    outputs: dict[str, Any], targets: list[dict[str, Any]], indices: Any
) -> list[dict[str, float]]:
    """Compare soft-skeleton mass at K=10/25/50/75 for all matched stem queries.

    K=75 is a numerical reference, not asserted to be a true converged skeleton. The result is a soft-mass ratio, not
    binary GT centerline coverage.
    """
    result = []
    pred = outputs["pred_masks"]
    for bi, (source, target) in enumerate(indices):
        source, target = source.cuda(), target.cuda()
        keep = targets[bi]["labels"][target] == 1
        if "valid" in targets[bi]:
            keep &= targets[bi]["valid"][target]
        source, target = source[keep], target[keep]
        masks = targets[bi]["masks"][target]
        height, width = masks.shape[-2:]
        padding = outputs["mask_padding"][bi]
        keep = (masks.bool() & ~padding[:height, :width]).flatten(1).any(1)
        source = source[keep]
        if not len(source):
            continue
        if isinstance(pred, Tensor):
            logits = pred[bi, source]
        else:
            logits = (
                torch.einsum("chw,nc->nhw", pred["spatial_features"][bi], pred["query_features"][bi, source])
                + pred["bias"]
            )
        for chunk in logits.split(8):  # type: ignore[no-untyped-call]
            probabilities = F.interpolate(
                chunk.float()[:, None], size=padding.shape, mode="bilinear", align_corners=False
            )
            probabilities = (
                probabilities[:, :, :height, :width].masked_fill(padding[:height, :width], -torch.inf).sigmoid()
            )
            mass = {k: _soft_skeleton(probabilities, k).flatten(1).sum(1).cpu().tolist() for k in (10, 25, 50, 75)}
            for j in range(len(chunk)):
                result.append({f"mass_k{k}": mass[k][j] for k in mass})
    return result


def compare_gradients(module: Any, dataset: CocoDetection, other: Any, args: Any, *, parity: bool) -> None:
    """Compare loss and full-model gradients on one shared forward/matching graph."""
    label = "equivalence" if parity else "k25_gradients"
    named = [(name, param) for name, param in module.model.named_parameters() if param.requires_grad]
    params = [param for _, param in named]
    groups = {
        name: [i for i, (key, _) in enumerate(named) if group_name(key) == name]
        for name in sorted({group_name(key) for key, _ in named})
    }
    groups["all"] = list(range(len(named)))
    rows, prediction_rows = [], []
    for bi in range(args.batches):
        samples, targets, meta = batch_at(dataset, bi, module.model_config.patch_size * module.model_config.num_windows)
        matched: dict[str, Any] = {}
        new_loss = module.criterion.loss_mask_cldice

        def capture(outputs: Any, truth: Any, indices: Any) -> Any:
            matched["indices"] = indices
            return new_loss(outputs, truth, indices)

        if parity:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(7000 + bi)
        with patch("rfdetr.models.criterion._soft_skeleton", wraps=_soft_skeleton) as spy:
            with patch.object(module.criterion, "loss_mask_cldice", side_effect=capture):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = module.model(samples, targets)
                    outputs["mask_padding"] = meta.pop("padding")
                    losses = module.criterion(outputs, targets)
            unique_gt = sum(call.args[0].shape[0] for call in spy.call_args_list)
            rng_before = torch.random.get_rng_state()
            cuda_rng_before = torch.cuda.get_rng_state()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                alternate = 0.1 * other.loss_mask_cldice(outputs, targets, matched["indices"])["loss_mask_cldice"]
            current = 0.1 * losses["loss_mask_cldice"]
            calls_before = spy.call_count
            native = torch.stack(
                [
                    v * module.criterion.weight_dict[k]
                    for k, v in losses.items()
                    if k in module.criterion.weight_dict and k != "loss_mask_cldice"
                ]
            ).sum()
            boundary_difference = None
            if parity:
                pred = outputs["pred_masks"]
                boundary = [pred] if isinstance(pred, Tensor) else list(pred.values())
                boundary_first = cpu_grads(current, boundary, retain_graph=True)
                boundary_second = cpu_grads(alternate, boundary, retain_graph=True)
                boundary_difference = gradient_difference(boundary_first, boundary_second)
                for left, right in zip(boundary_first, boundary_second, strict=True):
                    if left is not None:
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                # The adapter's complete input Jacobian is now checked. CUDA grid_sample
                # in the unchanged shared network has no deterministic backward kernel.
                torch.use_deterministic_algorithms(False)
            original_grad = cpu_grads(native, params, retain_graph=True) if not parity else None
            first_grad = cpu_grads(current, params, retain_graph=True)
            first_repeat = cpu_grads(current, params, retain_graph=True) if parity else None
            second_grad = cpu_grads(alternate, params, retain_graph=parity)
            second_repeat = cpu_grads(alternate, params, retain_graph=False) if parity else None
            assert spy.call_count == calls_before, "GT skeleton was recomputed during backward"
            assert torch.equal(rng_before, torch.random.get_rng_state())
            assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state())
        difference = gradient_difference(first_grad, second_grad)
        if parity:
            torch.testing.assert_close(current, alternate, atol=0, rtol=0)
        row = dict(
            batch=bi,
            **meta,
            new_k10_loss=float(current.detach()),
            alternate_loss=float(alternate.detach()),
            unique_gt_skeletons=unique_gt,
            gradient=difference,
        )
        if parity:
            assert first_repeat is not None and second_repeat is not None
            row["mask_input_gradient"] = boundary_difference
            row["shared_parameter_repeatability"] = verify_gradient_parity(
                first_grad, first_repeat, second_grad, second_repeat
            )
        if original_grad is not None:
            row["groups"] = {
                name: dict(
                    k10_vs_k25=gradient_relation([first_grad[i] for i in ids], [second_grad[i] for i in ids]),
                    original_vs_k25=gradient_relation([original_grad[i] for i in ids], [second_grad[i] for i in ids]),
                )
                for name, ids in groups.items()
            }
            prediction_rows.extend(prediction_skeleton_mass(outputs, targets, matched["indices"]))
        rows.append(row)
        write_json(args.output / f"{label}.json", rows)
        if parity:
            assert row["mask_input_gradient"]["bitwise_equal"]
        print(
            json.dumps(
                dict(
                    stage=label,
                    batch=bi,
                    loss=float(current.detach()),
                    alternate=float(alternate.detach()),
                    max_gradient_error=difference["max_absolute_error"],
                )
            ),
            flush=True,
        )
        del (
            outputs,
            losses,
            native,
            current,
            alternate,
            first_grad,
            second_grad,
            original_grad,
            first_repeat,
            second_repeat,
        )
    if not parity:
        write_json(args.output / "prediction_skeleton_mass.json", prediction_rows)


@torch.no_grad()
def coverage_k25(dataset: CocoDetection, args: Any) -> None:
    """Replay the 2,669 prior augmented GT masks and verify K=25 against extinction."""
    records: list[dict[str, Any]] = sum(
        [json.loads((args.coverage_records / f"coverage_seed{seed}.json").read_text()) for seed in (42, 43, 44)], []
    )
    result: list[dict[str, Any]] = []
    pending: list[Tensor] = []
    metadata: list[dict[str, Any]] = []

    def flush() -> None:
        masks = torch.stack(pending).cuda()
        _, reference, _, extinct = skeleton_trace(masks)
        candidate = _soft_skeleton(masks.float()[:, None], 25)[:, 0]
        counts = reference.flatten(1).sum(1)
        ratios = ((candidate > 0) & (reference > 0)).flatten(1).sum(1) / counts.clamp_min(1)
        for i, row in enumerate(metadata):
            result.append(
                dict(
                    seed=row["seed"],
                    dataset_index=row["dataset_index"],
                    target_index=row["target_index"],
                    old_k10_coverage=row["coverage"],
                    k25_coverage=float(ratios[i]),
                    k25_pixels=int(candidate[i].sum()),
                    reference_pixels=int(counts[i]),
                    extinct=bool(extinct[i]),
                )
            )
        pending.clear()
        metadata.clear()

    key, targets = None, None
    for row in records:
        current = (row["seed"], row["dataset_index"])
        if current != key:
            seed_sample(*current)
            _, targets = dataset[current[1]]
            key = current
        assert targets is not None
        mask = targets["masks"][row["target_index"]]
        assert int(mask.sum()) == row["area"] and list(mask.shape) == row["shape"]
        pending.append(mask)
        metadata.append(row)
        if len(pending) == 24:
            flush()
    if pending:
        flush()
    assert all(r["extinct"] and r["reference_pixels"] > 0 and r["k25_coverage"] == 1 for r in result)
    write_json(args.output / "coverage_k25.json", result)
    print(json.dumps(dict(stage="coverage_k25", instances=len(result), fully_covered=len(result))), flush=True)


def benchmark(module: Any, dataset: CocoDetection, old: Any, k25: Any, args: Any) -> None:
    """Time complete forward+loss+one backward, without optimizer steps.

    Warm every mode/shape, then rotate execution order. Report allocated memory; reserved allocator memory is not a fair
    live-activation comparison.
    """
    modes = {"legacy_k10": old, "dedup_k10": module.criterion, "dedup_k25": k25}
    rows = []
    for bi in range(args.batches):
        samples, targets, meta = batch_at(dataset, bi, module.model_config.patch_size * module.model_config.num_windows)
        for repeat in range(3):
            keys = list(modes)
            rotation = (bi + repeat) % len(keys)
            for name in keys[rotation:] + keys[:rotation]:
                module.model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(7000 + bi)
                begin = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = module.model(samples, targets)
                    outputs["mask_padding"] = meta["padding"]
                    losses = modes[name](outputs, targets)
                    total = torch.stack(
                        [v * modes[name].weight_dict[k] for k, v in losses.items() if k in modes[name].weight_dict]
                    ).sum()
                torch.autograd.backward(total)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                if repeat:
                    rows.append(
                        dict(
                            batch=bi,
                            scale=meta["scale"],
                            mode=name,
                            repeat=repeat,
                            seconds=elapsed,
                            loss=float(total.detach()),
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        )
                    )
                del outputs, losses, total
        write_json(args.output / "benchmark.json", rows)
        print(json.dumps(dict(stage="benchmark", batch=bi, measured_passes=len(rows))), flush=True)
    module.model.zero_grad(set_to_none=True)


def parameter_hash(module: Any) -> str:
    """Verify parameters stayed fixed without writing diagnostic checkpoints."""
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    """Run the parity gate, K=25 coverage/gradient probes, and bounded timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-criterion", type=Path, required=True)
    parser.add_argument("--coverage-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    mc, tc = configurations(args)
    assert tc.mask_cldice_loss_coef == 0.1 and tc.mask_cldice_class_ids == [1] and tc.mask_cldice_iterations == 10
    paths = [
        Path(__file__),
        args.config,
        args.checkpoint,
        args.legacy_criterion,
        Path(criterion_module.__file__),
        Path(criterion_module.__file__).with_name("soft_cldice.py"),
    ]
    assert tc.dataset_dir is not None
    paths.extend(Path(tc.dataset_dir) / split / "_annotations.coco.json" for split in ("train", "valid"))
    paths.extend(args.coverage_records / f"coverage_seed{seed}.json" for seed in (42, 43, 44))
    paths.append(Path(__file__).with_name("diagnose_tomato_e1.py"))
    write_json(
        args.output / "provenance.json",
        dict(
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            torch=torch.__version__,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            bf16=True,
            deterministic_mask_input_parity=True,
            deterministic_shared_backward=False,
            deterministic_benchmark=torch.are_deterministic_algorithms_enabled(),
            coefficient=0.1,
            iterations={"legacy": 10, "deduplicated": [10, 25], "prediction_mass_reference": [50, 75]},
            class_ids=[1],
            group_detr=mc.group_detr,
            optimizer_steps=0,
        ),
    )
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
    before = parameter_hash(module.model)
    write_json(args.output / "parameter_names.json", [n for n, p in module.model.named_parameters() if p.requires_grad])
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    dataset = data._dataset_train
    assert isinstance(dataset, CocoDetection)
    old = legacy_criterion(args.legacy_criterion, module.criterion)
    k25 = copy.copy(module.criterion)
    k25.mask_cldice_iterations = 25
    begin = time.monotonic()
    # Exact parity at every input of the modified adapter implies equal shared-parameter
    # derivatives by the chain rule. Record stochastic CUDA backward separately, then
    # restore training kernels before K=25 probes and timing.
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        compare_gradients(module, dataset, old, args, parity=True)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    write_json(
        args.output / "equivalence_passed.json",
        dict(
            passed=True, batches=args.batches, scope="exact_loss_and_all_mask_input_gradients", shared_backward="noisy"
        ),
    )
    coverage_k25(dataset, args)
    compare_gradients(module, dataset, k25, args, parity=False)
    benchmark(module, dataset, old, k25, args)
    after = parameter_hash(module.model)
    assert before == after
    write_json(
        args.output / "completed.json",
        dict(
            complete=True,
            optimizer_steps=0,
            parameters_unchanged=True,
            parameter_sha256=after,
            seconds=time.monotonic() - begin,
        ),
    )


if __name__ == "__main__":
    main()
