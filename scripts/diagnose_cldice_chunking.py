# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Check K=25 chunking against frozen unchunked code, then replay the previous OOM batch.

No optimizer is created. Real-logit parity uses the separability of the original per-instance loss to stream its
reference derivatives; this avoids repeating an unbounded reference backward on the previously failing batch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor

from rfdetr.datasets.coco import CocoDetection
from rfdetr.models import criterion as criterion_module
from rfdetr.models.soft_cldice import _soft_skeleton
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from scripts.diagnose_cldice_dedup import batch_at, gradient_difference, legacy_criterion, parameter_hash
from scripts.diagnose_tomato_e1 import configurations, group_name, write_json


def real_logits(module: Any, dataset: CocoDetection, reference: Any, bi: int) -> tuple[Any, Any, Any, Any]:
    """Capture native final-layer matches and reconstruct the same selected BF16 logits."""
    samples, targets, meta = batch_at(dataset, bi, module.model_config.patch_size * module.model_config.num_windows)
    matched: dict[str, Any] = {}
    original = reference.loss_mask_cldice

    def capture(outputs: Any, truth: Any, indices: Any) -> Any:
        matched["indices"] = indices
        return original(outputs, truth, indices)

    torch.manual_seed(7000 + bi)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module.model(samples, targets)
        outputs["mask_padding"] = meta["padding"]
        with patch.object(reference, "loss_mask_cldice", side_effect=capture):
            reference(outputs, targets)
        pred = outputs["pred_masks"]
        logits, remapped = [], []
        for i, (source, target) in enumerate(matched["indices"]):
            truth = targets[i]
            source, target = source.to("cuda"), target.to("cuda")
            selected = truth["labels"][target] == 1
            if "valid" in truth:
                selected &= truth["valid"][target]
            source, target = source[selected], target[selected]
            height, width = truth["masks"].shape[-2:]
            masks = truth["masks"][target].bool() & ~meta["padding"][i, :height, :width]
            nonempty = masks.flatten(1).any(1)
            source, target = source[nonempty], target[nonempty]
            value = (
                pred[i, source]
                if isinstance(pred, Tensor)
                else torch.einsum("chw,nc->nhw", pred["spatial_features"][i], pred["query_features"][i, source])
                + pred["bias"]
            )
            logits.append(value.float())
            remapped.append((torch.arange(len(source), device="cuda"), target))
        dense = logits[0].new_zeros((len(logits), max(len(z) for z in logits), *logits[0].shape[-2:]))
        for i, value in enumerate(logits):
            dense[i, : len(value)] = value
    return dense, targets, remapped, meta


def verify_batch(module: Any, dataset: CocoDetection, reference: Any, bi: int) -> dict[str, Any]:
    """Compare the whole scalar and every logit derivative with streamed original derivatives."""
    dense, targets, indices, meta = real_logits(module, dataset, reference, bi)
    count = sum(len(source) for source, _ in indices)
    assert count > 0
    rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        outputs = dict(pred_masks=dense, mask_padding=meta["padding"])
        with torch.no_grad():
            full_loss = 0.1 * reference.loss_mask_cldice(outputs, targets, indices)["loss_mask_cldice"]
        expected = torch.zeros_like(dense, device="cpu")
        streamed_loss = 0.0
        for i, (source, target) in enumerate(indices):
            if not len(source):
                continue
            height, width = targets[i]["masks"].shape[-2:]
            masks = targets[i]["masks"][target].bool() & ~meta["padding"][i, :height, :width]
            unique, inverse = torch.unique(target, return_inverse=True)
            with torch.no_grad():
                unique_masks = targets[i]["masks"][unique].bool() & ~meta["padding"][i, :height, :width]
                skeletons = _soft_skeleton(unique_masks.float()[:, None], 25)[:, 0]
            for j in range(len(source)):
                leaf = dense[i, j : j + 1].detach().requires_grad_()
                loss = (
                    reference._cldice_on_image(
                        leaf, masks[j : j + 1], meta["padding"][i], skeletons, inverse[j : j + 1]
                    )
                    / count
                    * 0.1
                )
                expected[i, j] = torch.autograd.grad(loss, leaf)[0][0].cpu()
                streamed_loss += float(loss.detach())
        leaf = dense.detach().requires_grad_()
        outputs["pred_masks"] = leaf
        loss = 0.1 * module.criterion.loss_mask_cldice(outputs, targets, indices)["loss_mask_cldice"]
        actual = torch.autograd.grad(loss, leaf)[0].cpu()
        torch.testing.assert_close(loss, full_loss, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(full_loss, full_loss.new_tensor(streamed_loss), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        difference = gradient_difference([expected], [actual])
        assert difference["relative_l2_error"] <= 1e-5
        assert torch.equal(rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        return dict(
            batch=bi,
            scale=meta["scale"],
            dataset_indices=meta["dataset_indices"],
            matches=count,
            matches_per_image=[len(source) for source, _ in indices],
            unchunked_loss=float(full_loss),
            streamed_reference_loss=streamed_loss,
            chunked_loss=float(loss.detach()),
            gradient=difference,
            rng_unchanged=True,
            passed=True,
        )
    finally:
        torch.use_deterministic_algorithms(deterministic)


def benchmark(module: Any, dataset: CocoDetection, reference: Any, output: Path) -> list[dict[str, Any]]:
    """Measure identical complete forwards and one native weighted backward, without optimizer steps."""
    modes = {"unchunked_k25": reference, "chunk4_k25": module.criterion}
    rows = []

    def run_step(criterion: Any, samples: Any, targets: Any, meta: Any, bi: int) -> dict[str, Any]:
        module.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(7000 + bi)
        begin = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module.model(samples, targets)
            outputs["mask_padding"] = meta["padding"]
            losses = criterion(outputs, targets)
            total = torch.stack(
                [v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict]
            ).sum()
        torch.autograd.backward(total)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated() / 2**20
        norms: dict[str, float] = {}
        for name, parameter in module.model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                group = group_name(name)
                norms[group] = norms.get(group, 0.0) + float(parameter.grad.float().square().sum())
        assert all(norms[name] > 0 for name in ("backbone", "transformer", "mask_head"))
        return dict(
            status="ok",
            seconds=seconds,
            loss=float(total.detach()),
            topology_loss=float(losses["loss_mask_cldice"].detach()) * 0.1,
            peak_allocated_mib=peak,
            gradient_squared_norms=norms,
        )

    for bi in range(3):
        samples, targets, meta = batch_at(dataset, bi, module.model_config.patch_size * module.model_config.num_windows)
        failed = set()
        for repeat in range(3):
            names = list(modes)
            if (bi + repeat) % 2:
                names.reverse()
            for name in names:
                if name in failed:
                    continue
                try:
                    result = run_step(modes[name], samples, targets, meta, bi)
                except torch.OutOfMemoryError as error:
                    result = dict(
                        status="oom", error=str(error), peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20
                    )
                    failed.add(name)
                rows.append(dict(batch=bi, scale=meta["scale"], mode=name, repeat=repeat, warmup=repeat == 0, **result))
                write_json(output / "benchmark.json", rows)
                if result["status"] == "oom":
                    module.model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
        print(json.dumps(dict(stage="benchmark", batch=bi, failed_modes=sorted(failed))), flush=True)
    module.model.zero_grad(set_to_none=True)
    return rows


def main() -> None:
    """Require parity before measuring capacity; retain source/data hashes and unchanged parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "prior-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-commit", default="b17a879dc46b900956b29891deb9847a879a68b4")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    mc, tc = configurations(args)
    assert tc.mask_cldice_loss_coef == 0.1 and tc.mask_cldice_class_ids == [1]
    assert tc.mask_cldice_iterations == 10 and tc.batch_size == 8 and mc.group_detr == 13
    tc.mask_cldice_iterations = 25
    tc.mask_cldice_chunk_size = 4
    prior = json.loads((args.prior_run / "provenance.json").read_text())
    criterion_path = Path(criterion_module.__file__)
    for name, expected in prior["sha256"].items():
        if Path(name).resolve() != criterion_path.resolve():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    frozen = args.output / "reference_criterion.py"
    frozen.write_bytes(
        subprocess.check_output(["git", "show", f"{args.reference_commit}:src/rfdetr/models/criterion.py"])
    )
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == prior["sha256"][str(criterion_path)]
    paths = [Path(__file__), criterion_path, frozen, args.prior_run / "provenance.json"]
    write_json(
        args.output / "provenance.json",
        dict(
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            arguments={key: str(value) for key, value in vars(args).items()},
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            verified_prior_sha256=prior["sha256"],
            model_config=mc.model_dump(mode="json"),
            train_config=tc.model_dump(mode="json"),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            optimizer_steps=0,
            parity_scope="FP32 mask-logit derivatives with frozen BF16 predictions; streamed separable reference",
            benchmark_scope="single GPU, native weighted forward/backward; warmup plus two measured repeats",
        ),
    )
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
    before = parameter_hash(module.model)
    reference = legacy_criterion(frozen, module.criterion)
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    dataset = data._dataset_train
    assert isinstance(dataset, CocoDetection)
    old_batches = json.loads((args.prior_run / "equivalence.json").read_text())
    rows = []
    begin = time.monotonic()
    for bi in range(3):
        row = verify_batch(module, dataset, reference, bi)
        assert row["dataset_indices"] == old_batches[bi]["dataset_indices"]
        rows.append(row)
        write_json(args.output / "equivalence.json", rows)
        print(json.dumps(dict(stage="parity", **row)), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "equivalence_passed.json", dict(passed=True, batches=3, relative_l2_limit=1e-5))
    measurements = benchmark(module, dataset, reference, args.output)
    after = parameter_hash(module.model)
    assert before == after
    chunk_rows = [row for row in measurements if row["mode"] == "chunk4_k25"]
    assert len(chunk_rows) == 9 and all(row["status"] == "ok" for row in chunk_rows)
    write_json(
        args.output / "completed.json",
        dict(
            complete=True,
            optimizer_steps=0,
            parameters_unchanged=True,
            parameter_sha256=after,
            seconds=time.monotonic() - begin,
            all_chunked_batches_fit=True,
        ),
    )


if __name__ == "__main__":
    main()
