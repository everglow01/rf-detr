# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compare native classification, mask and clDice gradients at frozen shared decoder outputs.

Only training images are used. A value-preserving autograd boundary at the transformer output keeps both heads attached
to the same decoder feature tensor while preventing parameter gradients. These local feature derivatives are not AdamW
parameter updates. Native auxiliary/box losses and the mask spatial-feature path are outside this boundary's comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import make_collate_fn
from scripts.diagnose_tomato_e1 import configurations, seed_sample, write_json
from scripts.diagnose_tomato_mask_conflicts import attribute_mask_gradients, capture_mask_points, dense_geometry
from scripts.diagnose_tomato_supervision import mask_overlaps


def cut_decoder_output(output: tuple[Any, ...]) -> tuple[Any, ...]:
    """Keep every forward value while making decoder features the common gradient boundary."""
    assert isinstance(output[0], Tensor) and output[0].ndim == 4
    return (output[0].detach().requires_grad_(), *output[1:])


def relation(left: Tensor, right: Tensor) -> dict[str, float | None]:
    """Measure direction, size and projection without interpreting zero gradients as agreement."""
    a, b = left.double().flatten(), right.double().flatten()
    aa, bb, dot = float(a @ a), float(b @ b), float(a @ b)
    return dict(
        left_norm=math.sqrt(aa),
        right_norm=math.sqrt(bb),
        cosine=max(-1.0, min(1.0, dot / math.sqrt(aa * bb))) if aa > 0 and bb > 0 else None,
        right_to_left_norm=math.sqrt(bb / aa) if aa > 0 else None,
        right_projection_over_left=dot / aa if aa > 0 else None,
    )


def measures(cls: Tensor, mask: Tensor, topology: Tensor, logit_jacobian: Tensor) -> dict[str, Any]:
    """Compare weighted feature gradients and their hypothetical descent directions."""
    return dict(
        cls_mask=relation(cls, mask),
        cls_cldice=relation(cls, topology),
        mask_cldice=relation(mask, topology),
        cls_native_sum=relation(cls, cls + mask),
        cls_sum_with_cldice=relation(cls, cls + mask + topology),
        stem_logit_descent=dict(
            cls=float(-(cls.double() * logit_jacobian.double()).sum()),
            mask=float(-(mask.double() * logit_jacobian.double()).sum()),
            cldice=float(-(topology.double() * logit_jacobian.double()).sum()),
        ),
    )


def choose_cohort(previous: Path) -> list[dict[str, Any]]:
    """Pair every previously observed hard training GT with a distinct high-confidence, nearest-area control."""
    arms: dict[str, dict[tuple[int, ...], list[dict[str, Any]]]] = {}
    for arm in ("baseline", "beta1", "beta2"):
        indexed: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for row in json.loads((previous / f"{arm}-training.json").read_text()):
            key = tuple(row[k] for k in ("augmentation_seed", "dataset_index", "image_id", "target_index"))
            indexed.setdefault(key, []).append(row)
        arms[arm] = indexed
    base = arms["baseline"]
    hard = sorted(
        k
        for k in base
        if any(r["winner"]["score"] < 0.3 and r["winner"]["mask_iou"] >= 0.5 for arm in arms.values() for r in arm[k])
    )
    assert len(hard) == 8, "The frozen source diagnostic cohort changed"
    hard_images = {k[2] for k in hard}
    pool = [
        k
        for k in base
        if k[2] not in hard_images
        and all(r["winner"]["score"] >= 0.7 and r["winner"]["mask_iou"] >= 0.7 for arm in arms.values() for r in arm[k])
    ]
    result = []
    for pair, key in enumerate(hard):
        area = base[key][0]["gt_mask_pixels"]
        control = min(
            (k for k in pool if k[0] == key[0]), key=lambda k: (abs(math.log(base[k][0]["gt_mask_pixels"] / area)), k)
        )
        pool.remove(control)
        for role, item in (("hard", key), ("control", control)):
            result.append(
                dict(
                    pair=pair,
                    role=role,
                    augmentation_seed=item[0],
                    dataset_index=item[1],
                    image_id=item[2],
                    target_index=item[3],
                    gt_mask_pixels=base[item][0]["gt_mask_pixels"],
                )
            )
    return result


def feature_gradient(loss: Tensor, features: Tensor) -> Tensor:
    """Stop at the shared feature tensor and retain only its final-layer derivative on CPU."""
    gradient = torch.autograd.grad(loss, features, retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and not bool(gradient[:-1].any())
    return gradient[-1].float().cpu()


def benchmark_topology(
    criterion: Any,
    outputs: dict[str, Any],
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],
    features: Tensor,
    variants: list[tuple[str, str, float | None]],
) -> list[dict[str, Any]]:
    """Time loss construction and shared-feature backward, including GT preparation, on a retained frozen forward.

    Warm up each variant, rotate their order, and synchronize CUDA around wall time. Report allocated increments as well
    as absolute peaks; this excludes upstream model/parameter backward and cannot predict epoch throughput.
    """
    results = []
    original = criterion.mask_cldice_mode, criterion.mask_cldice_recall_beta
    try:
        for repeat in range(-1, 5):
            shift = max(repeat, 0) % len(variants)
            for name, mode, beta in variants[shift:] + variants[:shift]:
                criterion.mask_cldice_mode = mode
                criterion.mask_cldice_recall_beta = beta or 1.0
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = 0.1 * criterion.loss_mask_cldice(outputs, targets, indices)[criterion.mask_topology_loss_key]
                gradient = torch.autograd.grad(loss, features, retain_graph=True)[0]
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                assert torch.isfinite(gradient).all()
                if repeat >= 0:
                    results.append(
                        dict(
                            variant=name,
                            repeat=repeat,
                            seconds=seconds,
                            allocated_before_mib=before / 1024**2,
                            peak_allocated_mib=peak / 1024**2,
                            extra_peak_mib=(peak - before) / 1024**2,
                        )
                    )
                del gradient, loss
    finally:
        criterion.mask_cldice_mode, criterion.mask_cldice_recall_beta = original
    return results


def probe_batch(
    module: Any,
    dataset: CocoDetection,
    cases: list[dict[str, Any]],
    batch_index: int,
    output: Path,
    gt_centerline: bool = False,
    split_mask: bool = False,
) -> None:
    """Hold matching and forward features fixed across all loss probes and both beta choices."""
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    examples = []
    for case in cases:
        seed_sample(case["augmentation_seed"], case["dataset_index"])
        example = dataset[case["dataset_index"]]
        assert int(example[1]["image_id"].item()) == case["image_id"]
        assert int(example[1]["labels"][case["target_index"]]) == 1
        assert int(example[1]["masks"][case["target_index"]].sum()) == case["gt_mask_pixels"]
        examples.append(example)
    collate = make_collate_fn(block_size=module.model_config.patch_size * module.model_config.num_windows, pack=False)
    samples, targets = collate(examples)
    samples = samples.to("cuda")
    targets = [{k: v.cuda() if isinstance(v, Tensor) else v for k, v in target.items()} for target in targets]
    model, criterion = module.model, module.criterion
    assert not any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) for layer in model.modules())
    captured: dict[str, Tensor] = {}

    def boundary(_module: Any, _inputs: Any, result: tuple[Any, ...]) -> tuple[Any, ...]:
        cut = cut_decoder_output(result)
        captured["features"] = cut[0]
        return cut

    torch.manual_seed(9000 + batch_index)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(samples, targets)
    torch.manual_seed(9000 + batch_index)
    hook = model.transformer.register_forward_hook(boundary)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(samples, targets)
    finally:
        hook.remove()
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(reference[key], outputs[key], rtol=0, atol=0)
    for key in reference["pred_masks"]:
        torch.testing.assert_close(reference["pred_masks"][key], outputs["pred_masks"][key], rtol=0, atol=0)
    del reference
    features = captured["features"]
    outputs["mask_padding"] = samples.mask
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        indices = criterion.matcher(outputs, targets, group_detr=criterion.group_detr)
    normalizer = criterion.num_boxes_for_targets(outputs, targets)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cls_loss = criterion.loss_labels(outputs, targets, indices, normalizer, log=False)["loss_ce"]
        cls_loss = cls_loss * criterion.weight_dict["loss_ce"]
    grads = {"cls": feature_gradient(cls_loss, features)}
    logit_jacobian = feature_gradient(outputs["pred_logits"][..., 1].sum(), features)
    losses = dict(cls=float(cls_loss.detach()))
    attribution = []
    for seed in (420, 421, 422):
        torch.manual_seed(seed + batch_index * 1000)
        with (
            capture_mask_points() if split_mask else nullcontext() as captured_points,
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            mask_losses = criterion.loss_masks(outputs, targets, indices, normalizer)
            loss = torch.stack([value * criterion.weight_dict[key] for key, value in mask_losses.items()]).sum()
        grads[f"mask_{seed}"] = feature_gradient(loss, features)
        losses[f"mask_{seed}"] = float(loss.detach())
        if split_mask:
            assert captured_points is not None
            attribution.append(
                attribute_mask_gradients(
                    mask_losses,
                    captured_points,
                    criterion,
                    features,
                    targets,
                    indices,
                    cases,
                    samples.mask,
                    seed,
                    grads,
                )
            )
    variants: list[tuple[str, str, float | None]] = [("cldice_1", "cldice", 1.0), ("cldice_2", "cldice", 2.0)]
    if gt_centerline:
        variants.append(("gt_centerline", "gt_centerline", None))
    for name, mode, beta in variants:
        criterion.mask_cldice_mode = mode
        criterion.mask_cldice_recall_beta = beta or 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = 0.1 * criterion.loss_mask_cldice(outputs, targets, indices)[criterion.mask_topology_loss_key]
        # Complete checkpoint recomputation before changing the criterion's beta attribute.
        grads[name] = feature_gradient(loss, features)
        losses[name] = float(loss.detach())
        del loss
    assert all(parameter.grad is None for parameter in module.parameters())
    positions, metadata = [], []
    for bi, (case, target, (sources, truth)) in enumerate(zip(cases, targets, indices, strict=True)):
        selected = sources[truth == case["target_index"]].tolist()
        assert len(selected) == criterion.group_detr
        per_image = dict(
            pred_logits=outputs["pred_logits"][bi : bi + 1],
            pred_masks={k: v if k == "bias" else v[bi : bi + 1] for k, v in outputs["pred_masks"].items()},
        )
        overlaps = mask_overlaps(per_image, target, tuple(samples.tensors.shape[-2:]))
        stem_ids = torch.where(target["labels"] == 1)[0].tolist()
        column = stem_ids.index(case["target_index"])
        for query in selected:
            positions.append((bi, query))
            metadata.append(
                dict(
                    **case,
                    batch_index=batch_index,
                    query=query,
                    group=query // module.model_config.num_queries,
                    score=float(outputs["pred_logits"][bi, query, 1].detach().sigmoid()),
                    mask_iou=float(overlaps[query, column]),
                )
            )
    bi, qi = torch.tensor(positions).T
    vectors = {key: value[bi, qi].numpy() for key, value in grads.items()}
    vectors["stem_logit_jacobian"] = logit_jacobian[bi, qi].numpy()
    np.savez_compressed(output / f"batch{batch_index}-gradients.npz", **vectors)
    rows = []
    summaries = []
    for name, _mode, beta in variants:
        for seed in (420, 421, 422):
            cls, mask, topology = grads["cls"], grads[f"mask_{seed}"], grads[name]
            summaries.append(
                dict(
                    variant=name,
                    beta=beta,
                    mask_seed=seed,
                    scope="all_final_queries",
                    metrics=measures(cls, mask, topology, logit_jacobian),
                )
            )
            summaries.append(
                dict(
                    variant=name,
                    beta=beta,
                    mask_seed=seed,
                    scope="selected_stem_queries",
                    metrics=measures(cls[bi, qi], mask[bi, qi], topology[bi, qi], logit_jacobian[bi, qi]),
                )
            )
            for j, (b, q) in enumerate(positions):
                rows.append(
                    dict(
                        **metadata[j],
                        variant=name,
                        beta=beta,
                        mask_seed=seed,
                        metrics=measures(cls[b, q], mask[b, q], topology[b, q], logit_jacobian[b, q]),
                    )
                )
    geometry: list[dict[str, Any]] = []
    if split_mask:
        geometry, crops = dense_geometry(outputs, targets, indices, cases, samples.mask, samples.tensors)
        np.savez_compressed(output / f"batch{batch_index}-crops.npz", **crops)
    probe_peak = torch.cuda.max_memory_allocated() / 1024**3
    performance = (
        benchmark_topology(criterion, outputs, targets, indices, features, variants)
        if gt_centerline and not split_mask
        else []
    )
    write_json(
        output / f"batch{batch_index}.json",
        dict(
            rows=rows,
            summaries=summaries,
            losses=losses,
            metadata=metadata,
            input_shape=list(samples.tensors.shape),
            feature_shape=list(features.shape),
            num_boxes=float(normalizer),
            cldice_selected_count=sum(int((t["labels"] == 1).sum()) for t in targets) * criterion.group_detr,
            weight_dict=criterion.weight_dict,
            forward_bitwise_equal=True,
            parameter_gradients=False,
            seconds=time.monotonic() - started,
            peak_allocated_gib=probe_peak,
            performance=performance,
            attribution=attribution,
            geometry=geometry,
        ),
    )
    print(f"{output.name}: batch={batch_index} shared-feature probes complete", flush=True)


def main() -> None:
    """Run the fixed hard/control training cohort and preserve checkpoint hashes and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--gt-centerline", action="store_true", help="Also compare GT-only coverage and loss performance"
    )
    parser.add_argument("--split-mask", action="store_true", help="Attribute native CE/Dice gradients by point region")
    args = parser.parse_args()
    if args.split_mask:
        args.gt_centerline = True
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert json.loads((args.previous / "completed.json").read_text())["complete"]
    cohort = choose_cohort(args.previous)
    checkpoints = json.loads((args.previous / "protocol.json").read_text())["checkpoints"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            git_commit=args.expected_commit,
            cohort=cohort,
            checkpoints=checkpoints,
            optimizer_steps=0,
            parameter_backward=False,
            boundary="transformer returned hs; compare last decoder layer",
            batch_size=8,
            groups=13,
            input_scale=504,
            mask_sampling_seeds=[420, 421, 422],
            cldice=dict(iterations=25, coefficient=0.1, chunk_size=4, class_ids=[1], betas=[1, 2]),
            gt_centerline=args.gt_centerline,
            split_mask=args.split_mask,
            benchmark=dict(
                enabled=args.gt_centerline and not args.split_mask,
                warmup_per_variant=1,
                repeats=5,
                scope="loss forward and shared-feature backward",
            ),
            baseline_cldice="counterfactual only; baseline was trained without clDice",
            limitations=(
                "Post-hoc training diagnostic cohort, correlated groups; no DDP/AdamW replay; "
                "no auxiliary/box or spatial-backbone gradient comparison."
            ),
        ),
    )
    torch.set_num_threads(2)
    for row in checkpoints:
        checkpoint = Path(row["path"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
        output = args.output / row["arm"]
        output.mkdir()
        config = checkpoint.parent.parent / ("training.yaml" if row["arm"] == "baseline" else f"{row['arm']}.yaml")
        mc, tc = configurations(argparse.Namespace(config=config, checkpoint=checkpoint, output=output))
        tc.amp_dtype = None
        module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().train()
        module.requires_grad_(False)
        criterion = module.criterion
        criterion.mask_cldice_class_ids = (1,)
        criterion.mask_cldice_iterations = 25
        criterion.mask_cldice_chunk_size = 4
        data = RFDETRDataModule(mc, tc)
        data.setup("fit")
        dataset = data._dataset_train
        assert isinstance(dataset, CocoDetection)
        for start in range(0, len(cohort), 8):
            probe_batch(
                module, dataset, cohort[start : start + 8], start // 8, output, args.gt_centerline, args.split_mask
            )
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
        del module, data
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        args.output / "completed.json",
        dict(complete=True, checkpoint_hashes_unchanged=True, optimizer_steps=0, parameter_gradients=False),
    )
    print("SHARED_DECODER_GRADIENT_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
