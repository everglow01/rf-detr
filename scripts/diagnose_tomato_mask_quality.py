# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Measure matched mask-IoU supervision on frozen training examples and paired no-update backward passes."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import make_collate_fn
from scripts.diagnose_cldice_dedup import parameter_hash
from scripts.diagnose_tomato_decoder_gradients import cut_decoder_output, relation
from scripts.diagnose_tomato_e1 import configurations, seed_sample, write_json


def batch(rows: list[dict[str, Any]], dataset: Any, block: int) -> tuple[Any, list[dict[str, Tensor]]]:
    """Recreate archived training augmentations and return one batch on CUDA."""
    examples = []
    for row in rows:
        seed_sample(row["augmentation_seed"], row["dataset_index"])
        sample = dataset[row["dataset_index"]]
        assert int(sample[1]["image_id"].item()) == row["image_id"]
        examples.append(sample)
    samples, targets = make_collate_fn(block_size=block, pack=False)(examples)
    if len(rows) == 1:
        digest = hashlib.sha256(
            samples.tensors.numpy().tobytes()
            + targets[0]["masks"].numpy().tobytes()
            + targets[0]["labels"].numpy().tobytes()
        ).hexdigest()
        assert digest == rows[0]["input_sha256"]
    return samples.to("cuda"), [{k: v.cuda() if isinstance(v, Tensor) else v for k, v in t.items()} for t in targets]


def probe(module: Any, row: dict[str, Any], samples: Any, targets: list[dict[str, Tensor]]) -> list[dict[str, Any]]:
    """Compare final-layer gradients at the classifier and its shared decoder input on identical matches."""
    captured: list[Tensor] = []

    def boundary(_module: Any, _args: Any, output: tuple[Any, ...]) -> tuple[Any, ...]:
        cut = cut_decoder_output(output)
        captured.append(cut[0])
        return cut

    handle = module.model.transformer.register_forward_hook(boundary)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module.model(samples)
    finally:
        handle.remove()
    outputs["mask_padding"] = samples.mask
    criterion = module.criterion
    features = captured[0]
    logits = outputs["pred_logits"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        matches = criterion.matcher(outputs, targets, group_detr=1)
        denominator = criterion.num_boxes_for_targets(outputs, targets)
        native = criterion.loss_labels(outputs, targets, matches, denominator, log=False)
        native.update(criterion.loss_masks(outputs, targets, matches, denominator))
        native.update(criterion.loss_boxes(outputs, targets, matches, denominator))
        cls = native["loss_ce"] * criterion.weight_dict["loss_ce"]
        total = torch.stack([native[k] * w for k, w in criterion.weight_dict.items() if k in native]).sum()
        aux = (
            criterion.loss_mask_quality(outputs, targets, matches)["loss_mask_quality"]
            * criterion.weight_dict["loss_mask_quality"]
        )
        qs, ts, ious = criterion._matched_mask_quality(outputs, 0, targets[0], matches[0])
    grad_quality, grad_logit = torch.autograd.grad(aux, (features, logits), retain_graph=True)
    grad_native = torch.autograd.grad(total, features, retain_graph=True)[0]
    grad_cls = torch.autograd.grad(cls, features, retain_graph=True)[0]
    jac = torch.autograd.grad(logits[..., 1].sum(), features)[0][-1, 0].float()
    results = []
    for q, t, iou in zip(qs.tolist(), ts.tolist(), ious.tolist(), strict=True):
        gt = next(g for g in row["stems"] if g["target_index"] == t)
        score = float(logits[0, q, 1].detach().float().sigmoid())
        raw_good = [c for c in gt["candidates"] if c["mask_iou"] >= 0.5]
        missed_direct = [c["query"] for c in raw_good if c["query"] != q and float(grad_logit[0, c["query"], 1]) == 0]
        cls_f, native_f, aux_f = (x[-1, 0, q].float() for x in (grad_cls, grad_native, grad_quality))
        expected = 0.1 * (score - iou) / len(qs)
        actual = float(grad_logit[0, q, 1])
        assert abs(actual - expected) <= 0.001
        results.append(
            dict(
                image=row["image_id"],
                seed=row["augmentation_seed"],
                gt=t,
                query=q,
                archived_low_good=gt["low_score_good"],
                score=score,
                iou=iou,
                quality_dlogit=actual,
                archived_match=next(c["query"] for c in gt["candidates"] if c["matched"]),
                unsupervised_archived_good_queries=missed_direct,
                cls_quality=relation(cls_f, aux_f),
                native_quality=relation(native_f, aux_f),
                cls_descent=float(-cls_f @ jac[q]),
                native_descent=float(-native_f @ jac[q]),
                quality_descent=float(-aux_f @ jac[q]),
                combined_descent=float(-(native_f + aux_f) @ jac[q]),
            )
        )
    return results


def measured_step(
    module: Any, samples: Any, targets: list[dict[str, Tensor]], padding: Tensor, enabled: bool, seed: int
) -> dict[str, Any]:
    """Time full model forward, matching, all native losses and parameter backward without updating weights."""
    weights = module.criterion.weight_dict
    if enabled:
        weights["loss_mask_quality"] = 0.1
    else:
        weights.pop("loss_mask_quality", None)
    module.model.mask_cldice_enabled = enabled
    module.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module.model(samples, targets)
        if enabled:
            outputs["mask_padding"] = padding
        losses = module.criterion(outputs, targets)
        loss = torch.stack([value * weights[key] for key, value in losses.items() if key in weights]).sum()
    torch.autograd.backward(loss)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return dict(
        enabled=enabled,
        ms=seconds * 1000,
        allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        native={k: float(v.detach()) for k, v in losses.items() if k in weights and k != "loss_mask_quality"},
    )


def benchmark(module: Any, dataset: Any, records: list[dict[str, Any]], block: int) -> list[dict[str, Any]]:
    """Warm each arm separately, reverse block order and pair forward randomness at two fixed scales."""
    # Original trainable-parameter flags are retained for the representative full backward.
    module.train()
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in module.model.modules())
    results = []
    for i, size in enumerate((384, 504)):
        stem = [r for r in records if r["role"] == "stem" and r["augmentation_seed"] == 42]
        background = [r for r in records if r["role"] == "background" and r["augmentation_seed"] == 42]
        rows = stem[6 * i : 6 * i + 6] + background[2 * i : 2 * i + 2]
        samples, targets = batch(rows, dataset, block)
        padding = samples.mask
        samples.tensors = F.interpolate(samples.tensors, (size, size), mode="bilinear", align_corners=False)
        samples.mask = F.interpolate(padding[:, None].float(), (size, size), mode="nearest")[:, 0].bool()
        for cycle in range(2):
            paired: dict[int, dict[str, float]] = {}
            for enabled in (bool(cycle), not bool(cycle)):
                module.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                measured_step(module, samples, targets, padding, enabled, 1100)
                for repeat in range(4):
                    item = measured_step(module, samples, targets, padding, enabled, 1200 + repeat)
                    reference = item.pop("native")
                    if repeat in paired:
                        assert reference == paired[repeat], "Paired native losses changed"
                    else:
                        paired[repeat] = reference
                    results.append(
                        dict(scale=size, cycle=cycle, repeat=repeat, images=[r["image_id"] for r in rows], **item)
                    )
            print("QUALITY_BENCH", size, cycle, flush=True)
    module.zero_grad(set_to_none=True)
    return results


def main() -> None:
    """Use an existing baseline checkpoint, training augmentations and previously identified low-score masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = args.short_run / "baseline/last_ema.pth"
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha == json.loads((args.short_run / "completed.json").read_text())["checkpoint_sha256"]["baseline"]
    assert json.loads((args.previous / "completed.json").read_text())["complete"]
    mc, tc = configurations(
        argparse.Namespace(config=args.short_run / "baseline.yaml", checkpoint=checkpoint, output=output)
    )
    tc.mask_quality_class_ids = [1]
    tc.mask_quality_loss_coef = 0.1
    tc.amp_dtype = None
    assert tc.mask_cldice_loss_coef == 0 and not mc.query_point_head
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().eval()
    before = parameter_hash(module.model)
    trainable = [p for p in module.parameters() if p.requires_grad]
    module.requires_grad_(False)
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    dataset = data._dataset_train
    assert isinstance(dataset, CocoDetection) and dataset.cat2label is not None
    assert any(dataset.coco.cats[c]["name"] == "stem" and label == 1 for c, label in dataset.cat2label.items())
    block = mc.patch_size * mc.num_windows
    records = json.loads((args.previous / "baseline-records.json").read_text())
    audits = {
        (r["image_id"], r["seed"]): r["sha256"]
        for r in json.loads((args.previous / "baseline-input-audit.json").read_text())
    }
    hard = [r for r in records if any(g["low_score_good"] for g in r["stems"])]
    hard_ids = {r["image_id"] for r in hard}
    controls = [
        r
        for r in records
        if r["image_id"] not in hard_ids
        and any(
            c["score"] >= 0.7 and c["matched"] and c["mask_iou"] >= 0.7 for g in r["stems"] for c in g["candidates"]
        )
    ]
    results = []
    for row in hard + controls[: len(hard)]:
        row["input_sha256"] = audits[row["image_id"], row["augmentation_seed"]]
        samples, targets = batch([row], dataset, block)
        results.extend(probe(module, row, samples, targets))
    write_json(output / "gradients.json", results)
    print("QUALITY_GRADIENTS", len(results), flush=True)
    for parameter in trainable:
        parameter.requires_grad_(True)
    metrics = benchmark(module, dataset, records, block)
    assert before == parameter_hash(module.model)
    assert sha == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(output / "benchmark.json", metrics)
    write_json(
        output / "completed.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            checkpoint_sha256=sha,
            parameter_hash=before,
            optimizer_steps=0,
            group_detr=mc.group_detr,
            gpu=torch.cuda.get_device_name(),
            coef=0.1,
            unchanged=True,
        ),
    )
    print("QUALITY_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
