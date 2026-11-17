# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Inspect frozen stem matching and IA-BCE supervision; never create an optimizer.

Validation uses the previously saved all-query geometry, verified against fresh logits. Training uses independent
augmented images, all Group DETR groups and native final-layer matching. These are checkpoint snapshots, not a replay of
historical training assignments. Gradients stop at detached logits; no model parameters receive gradients.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou, elementwise_box_iou
from rfdetr.utilities.tensors import make_collate_fn
from scripts.diagnose_tomato_e1 import configurations, seed_sample, write_json


def supervision(
    criterion: Any, outputs: dict[str, Any], target: dict[str, Tensor], indices: Any, gradient_probe: bool = True
) -> dict[str, Tensor]:
    """Read actual native IA-BCE logit gradients and reconstruct its detached positive targets.

    Positive gradients lower a logit under gradient descent. A negative stem gradient raises it. Gradients here use the
    native classification coefficient and group normalizer, without an optimizer or backbone graph.
    """
    assert criterion.ia_bce_loss
    logits = outputs["pred_logits"].detach().clone().requires_grad_(gradient_probe)
    boxes = outputs["pred_boxes"].detach()
    normalizer = criterion.num_boxes_for_targets(outputs, [target])
    if gradient_probe:
        with torch.autocast(logits.device.type, dtype=torch.bfloat16, enabled=logits.is_cuda):
            loss = criterion.loss_labels(
                dict(pred_logits=logits, pred_boxes=boxes), [target], indices, normalizer, log=False
            )
            gradient = torch.autograd.grad(loss["loss_ce"] * criterion.weight_dict["loss_ce"], logits)[0]
    q, t = (v.to(logits.device) for v in indices[0])
    assigned = torch.full((logits.shape[1],), -1, device=logits.device, dtype=torch.long)
    assigned[q] = t
    labels = torch.full_like(assigned, -1)
    labels[q] = target["labels"][t]
    ious = box_iou(box_cxcywh_to_xyxy(boxes[0].float()), box_cxcywh_to_xyxy(target["boxes"].float()))[0]
    probabilities = logits.detach()[0].sigmoid()
    soft_targets = torch.zeros_like(probabilities)
    matched_ious, _ = elementwise_box_iou(box_cxcywh_to_xyxy(boxes[0, q]), box_cxcywh_to_xyxy(target["boxes"][t]))
    values = probabilities[q, labels[q]].pow(criterion.focal_alpha) * matched_ious.pow(1 - criterion.focal_alpha)
    soft_targets[q, labels[q]] = values.clamp(min=0.01).to(soft_targets.dtype)
    if not gradient_probe:
        # Validation is observation only: describe the hypothetical IA-BCE direction algebraically.
        p = probabilities.float()
        gradient = p**3 + 2 * p**2 * (1 - p) * F.softplus(logits.detach()[0].float())
        gradient[q, labels[q]] = p[q, labels[q]] - soft_targets[q, labels[q]].float()
        gradient = (gradient * criterion.weight_dict["loss_ce"] / normalizer).unsqueeze(0)
    return dict(
        scores=probabilities.float(),
        assigned=assigned,
        assigned_class=labels,
        box_iou=ious,
        soft_targets=soft_targets.float(),
        gradient=gradient[0].float(),
        normalizer=normalizer,
    )


def capture_match(
    matcher: Any, outputs: dict[str, Any], target: dict[str, Tensor], groups: int
) -> tuple[Any, Tensor, Tensor]:
    """Capture the exact native solver input without drawing a second random mask sample."""
    captured = []
    assign = matcher._assign_compact_cost_matrix

    def capture(cost: Tensor, sizes: list[int], group_detr: int) -> Any:
        captured.append(cost.detach().clone())
        return assign(cost, sizes, group_detr)

    with patch.object(matcher, "_assign_compact_cost_matrix", side_effect=capture):
        indices = matcher(outputs, [target], group_detr=groups)
    assert len(captured) == 1, "This diagnostic requires the batch-one native solver path"
    classification = matcher.cost_class * matcher._focal_classification_cost(
        outputs["pred_logits"][0, :, target["labels"]]
    )
    return indices, captured[0], classification.detach().float().cpu()


@torch.no_grad()
def mask_overlaps(outputs: dict[str, Any], target: dict[str, Tensor], padded_size: tuple[int, int]) -> Tensor:
    """Measure binary mask IoU at the augmented GT canvas, one small query chunk at a time."""
    stem = torch.where(target["labels"] == 1)[0]
    truth = target["masks"][stem].float().flatten(1)
    masks = outputs["pred_masks"]
    pieces = []
    for start in range(0, outputs["pred_logits"].shape[1], 16):
        if isinstance(masks, Tensor):
            logits = masks[0, start : start + 16]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = torch.einsum(
                    "chw,nc->nhw", masks["spatial_features"][0], masks["query_features"][0, start : start + 16]
                )
                logits = logits + masks["bias"]
        predicted = (
            (
                F.interpolate(logits.float()[:, None], size=padded_size, mode="bilinear", align_corners=False)[
                    :, 0, : target["masks"].shape[-2], : target["masks"].shape[-1]
                ]
                > 0
            )
            .float()
            .flatten(1)
        )
        intersection = predicted @ truth.T
        union = predicted.sum(1)[:, None] + truth.sum(1)[None, :] - intersection
        pieces.append(intersection / union.clamp(min=1))
    return torch.cat(pieces).cpu()


def records_for_image(
    outputs: dict[str, Any],
    target: dict[str, Tensor],
    criterion: Any,
    overlaps: Tensor,
    groups: int,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Distinguish orphan low-score candidates from ordinary suppressed duplicates, separately in every group."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        indices, total_cost, class_cost = capture_match(criterion.matcher, outputs, target, groups)
    state = {
        k: v.cpu()
        for k, v in supervision(criterion, outputs, target, indices, gradient_probe=meta["split"] == "train").items()
    }
    meta = dict(meta, gradient_source="native_logit_autograd" if meta["split"] == "train" else "hypothetical_analytic")
    stem_indices = torch.where(target["labels"].cpu() == 1)[0].tolist()
    group_size = outputs["pred_logits"].shape[1] // groups
    rows = []

    def describe(query: int, ti: int, column: int) -> dict[str, Any]:
        grad = float(state["gradient"][query, 1])
        assigned = int(state["assigned"][query])
        return dict(
            query=query,
            score=float(state["scores"][query, 1]),
            mask_iou=float(overlaps[query, column]),
            box_iou=float(state["box_iou"][query, ti]),
            assigned_target=assigned,
            assigned_class=int(state["assigned_class"][query]),
            matched_same_gt=assigned == ti,
            stem_target=float(state["soft_targets"][query, 1]),
            gradient=grad,
            direction="raise" if grad < 0 else "lower" if grad > 0 else "zero",
            class_cost=float(class_cost[query, ti]),
            total_cost=float(total_cost[query, ti]),
            geometry_cost=float(total_cost[query, ti] - class_cost[query, ti]),
        )

    for column, ti in enumerate(stem_indices):
        for group in range(groups):
            queries = torch.arange(group * group_size, (group + 1) * group_size)
            fit = queries[overlaps[queries, column] >= 0.5]
            winners = queries[state["assigned"][queries] == ti]
            assert len(winners) == 1
            winner = int(winners[0])
            best = int(queries[overlaps[queries, column].argmax()])
            candidate = int(fit[state["scores"][fit, 1].argmax()]) if len(fit) else best
            low = fit[state["scores"][fit, 1] < 0.3]
            rows.append(
                dict(
                    **meta,
                    target_index=ti,
                    gt_mask_pixels=int(target["masks"][ti].sum()),
                    group=group,
                    fit_candidates=len(fit),
                    low_fit_candidates=len(low),
                    low_fit_same_gt=int((state["assigned"][low] == ti).sum()),
                    low_fit_negative_stem=int((state["assigned_class"][low] != 1).sum()),
                    orphan_low_score=bool(len(fit) and state["scores"][fit, 1].max() < 0.3),
                    candidate=describe(candidate, ti, column),
                    winner=describe(winner, ti, column),
                    best_mask=describe(best, ti, column),
                    normalizer=float(state["normalizer"]),
                )
            )
    return rows


def run_arm(args: argparse.Namespace, arm: str, checkpoint: Path, config: Path) -> None:
    """Inspect validation candidates and a predeclared training sample using identical seeds across arms."""
    mc, tc = configurations(argparse.Namespace(config=config, checkpoint=checkpoint, output=args.output / arm))
    tc.batch_size = tc.eval_batch_size = 1
    tc.amp_dtype = None
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).cuda().eval()
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    previous = {r["image_id"]: r for r in json.loads((args.previous / "recall" / arm / "queries.json").read_text())}
    validation = []
    for samples, targets in data.val_dataloader():
        image_id = int(targets[0]["image_id"].item())
        prior = previous[image_id]
        if not prior["target_indices"]:
            continue
        target = {k: v.cuda() if isinstance(v, Tensor) else v for k, v in targets[0].items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module.model(samples.to("cuda"))
        torch.testing.assert_close(
            outputs["pred_logits"][0].sigmoid().float().cpu(), torch.tensor(prior["query_scores"]), rtol=0, atol=0
        )
        assert torch.where(target["labels"] == 1)[0].tolist() == prior["target_indices"]
        for seed in (42, 43, 44):
            torch.manual_seed(seed * 1000000 + image_id)
            validation.extend(
                records_for_image(
                    outputs,
                    target,
                    module.criterion,
                    torch.tensor(prior["mask_iou"]),
                    1,
                    dict(image_id=image_id, matching_seed=seed, split="valid"),
                )
            )
    write_json(args.output / f"{arm}-validation.json", validation)
    print(f"{arm}: validation snapshots={len(validation)}", flush=True)
    dataset = data._dataset_train
    assert isinstance(dataset, CocoDetection) and dataset.cat2label is not None
    stem_categories = [key for key, label in dataset.cat2label.items() if label == 1]
    candidates = [
        i
        for i, image_id in enumerate(dataset.ids)
        if any(a["category_id"] in stem_categories for a in dataset.coco.imgToAnns[image_id])
    ]
    selected = sorted(
        np.random.default_rng(20261117).choice(candidates, min(128, len(candidates)), replace=False).tolist()
    )
    write_json(
        args.output / f"{arm}-training-selection.json",
        dict(
            dataset_indices=selected, image_ids=[dataset.ids[i] for i in selected], transform=repr(dataset._transforms)
        ),
    )
    module.train()
    collate = make_collate_fn(block_size=mc.patch_size * mc.num_windows, pack=False)
    training = []
    for seed in (42, 43):
        for count, index in enumerate(selected, 1):
            seed_sample(seed, index)
            samples, targets = collate([dataset[index]])
            target = {k: v.cuda() if isinstance(v, Tensor) else v for k, v in targets[0].items()}
            if not bool((target["labels"] == 1).any()):
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = module.model(samples.to("cuda"), [target])
            overlaps = mask_overlaps(outputs, target, tuple(samples.tensors.shape[-2:]))
            training.extend(
                records_for_image(
                    outputs,
                    target,
                    module.criterion,
                    overlaps,
                    module.criterion.group_detr,
                    dict(
                        image_id=int(target["image_id"].item()),
                        dataset_index=index,
                        augmentation_seed=seed,
                        split="train",
                        input_shape=list(samples.tensors.shape),
                    ),
                )
            )
            if count % 16 == 0:
                print(f"{arm}: train seed={seed} images={count}/{len(selected)}", flush=True)
        write_json(args.output / f"{arm}-training.json", training)
    assert all(p.grad is None for p in module.parameters())
    del module, data, outputs
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """Preserve model/data provenance around a bounded, frozen diagnostic run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    provenance = json.loads((args.previous / "provenance.json").read_text())
    selected = [r for r in provenance["checkpoints"] if r["kind"] == "best"]
    assert {r["arm"] for r in selected} == {"baseline", "beta1", "beta2"}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    write_json(
        args.output / "protocol.json",
        dict(
            git_commit=args.expected_commit,
            checkpoints=selected,
            optimizer_steps=0,
            network_backward=False,
            batch_size=1,
            train_groups="native",
            training_images=128,
            augmentation_seeds=[42, 43],
            validation_matching_seeds=[42, 43, 44],
            loss="native final-layer IA-BCE only",
            limitations=(
                "Frozen EMA snapshots, not historical assignments; batch-one native augmentation, "
                "no DDP/batch multiscale replay; validation matching is hypothetical, not training supervision."
            ),
        ),
    )
    for row in selected:
        checkpoint = Path(row["path"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
        root = checkpoint.parent
        config = root.parent / "training.yaml" if row["arm"] == "baseline" else root.parent / f"{row['arm']}.yaml"
        torch.manual_seed(42)
        run_arm(args, row["arm"], checkpoint, config)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
    write_json(args.output / "completed.json", dict(complete=True, optimizer_steps=0, checkpoint_hashes_unchanged=True))
    print("MATCHING_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
