"""Re-evaluate frozen E0 checkpoints using the original RF-DETR validation path."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pycocotools import mask as mask_utils
from scipy import ndimage
from skimage.morphology import skeletonize

from rfdetr.config import RFDETRSegSmallConfig, SegmentationTrainConfig
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule


class DetailedMAP(OnePassCocoMeanAveragePrecision):
    """Read per-class AP50/AP75 from the same complete COCO precision grid."""

    def _reduce_per_class(self, evaluation, prefix, classes):
        result = super()._reduce_per_class(evaluation, prefix, classes)
        precision = np.asarray(evaluation['precision'])
        for threshold in (0.5, 0.75):
            ti = int(np.flatnonzero(np.isclose(self.iou_thresholds, threshold))[0])
            values = []
            for k in range(len(classes)):
                p = precision[ti, :, k, 0, -1]
                p = p[p > -1]
                values.append(float(p.mean()) if p.size else -1.)
            result[f'{prefix}map{int(threshold*100)}_per_class'] = torch.tensor(values)
        return result


def encode(mask):
    result = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    result['counts'] = result['counts'].decode()
    return result


def topology(pred, target):
    skp, skt = skeletonize(pred), skeletonize(target)
    precision = float((skp & target).sum() / max(1, skp.sum()))
    sensitivity = float((skt & pred).sum() / max(1, skt.sum()))
    cldice = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    counts = []
    for mask in (pred, target):
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        areas = np.bincount(labels.ravel())[1:]
        counts.append(int((areas >= 8).sum()))
    return cldice, counts[0], counts[1]


def run(checkpoint, label, root, limit, device):
    output = root / label
    output.mkdir(exist_ok=False)
    baseline = Path('/usr/local/rfdetr26/output/output1/baseline-seg-small-v1-seed42')
    cfg = yaml.safe_load((baseline / 'training.yaml').read_text())['model']
    mc_args = dict(cfg['model_config']['init_args'])
    mc_args.update(device=device, pretrain_weights=str(checkpoint))
    tc_args = dict(cfg['train_config']['init_args'])
    tc_args.update(accelerator='cpu', devices=1, strategy='auto', batch_size=1, eval_batch_size=1,
                   num_workers=0, amp_dtype=None, use_ema=False, compute_val_loss=False,
                   output_dir=str(output), tensorboard=False, progress_bar=None, pack_targets=False,
                   multi_scale='per-batch', augmentation_backend='torchvision')
    mc, tc = RFDETRSegSmallConfig(**mc_args), SegmentationTrainConfig(**tc_args)
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False).to(device).eval()
    data = RFDETRDataModule(mc, tc)
    data.setup('validate')
    loader = data.val_dataloader()
    converter = COCOEvalCallback(max_dets=500, segmentation=True, eval_backend='vernier')
    metric = DetailedMAP(iou_type=('bbox', 'segm'), class_metrics=True,
                         max_detection_thresholds=[1, 10, 500], backend='vernier', num_classes=3)
    records, image_records = [], []
    start = time.monotonic()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            samples, targets = batch
            batch = (samples.to(device), [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                                           for k, v in t.items()} for t in targets])
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device=='cuda'):
                outputs = module.validation_step(batch, index)
            preds = converter._convert_preds(outputs['results'])
            targets = converter._convert_targets(outputs['targets'], preds)
            metric.update(preds, targets)
            for p, t, raw_target in zip(preds, targets, outputs['targets']):
                image_id = int(raw_target['image_id'].reshape(-1)[0])
                pi = torch.where((p['labels'] == 1) & (p['scores'] >= 0.3))[0]
                pi = pi[torch.argsort(p['scores'][pi], descending=True)]
                gi = torch.where(t['labels'] == 1)[0]
                pm = p['masks'][pi].cpu().numpy().astype(bool)
                gm = t['masks'][gi].cpu().numpy().astype(bool)
                ious = np.zeros((len(pi), len(gi)))
                for j, pred_mask in enumerate(pm):
                    for k, gt_mask in enumerate(gm):
                        ious[j, k] = (pred_mask & gt_mask).sum() / max(1, (pred_mask | gt_mask).sum())
                matched = {}
                for j in range(len(pi)):
                    available = [k for k in range(len(gi)) if k not in matched]
                    if available:
                        k = max(available, key=lambda k: ious[j, k])
                        if ious[j, k] >= 0.5:
                            matched[k] = j
                for k, gt_mask in enumerate(gm):
                    j = int(ious[:, k].argmax()) if len(pi) else None
                    matched_j = matched.get(k)
                    selected_j = matched_j if matched_j is not None else j
                    pred_mask = pm[selected_j] if selected_j is not None else np.zeros_like(gt_mask)
                    cldice, pred_components, gt_components = topology(pred_mask, gt_mask)
                    records.append(dict(image_id=image_id, target_index=int(gi[k]), matched=matched_j is not None,
                        best_iou=float(ious[j, k]) if j is not None else 0.,
                        matched_iou=float(ious[matched_j, k]) if matched_j is not None else None,
                        cldice=cldice, pred_components=pred_components, gt_components=gt_components,
                        score=float(p['scores'][pi[selected_j]]) if selected_j is not None else 0.))
                image_records.append(dict(image_id=image_id, gt_stems=len(gi), predicted_stems=len(pi),
                                          matched=len(matched), false_positives=len(pi)-len(matched),
                                          gt_masks=[encode(m) for m in gm], pred_masks=[encode(m) for m in pm],
                                          scores=p['scores'][pi].tolist()))
            if (index + 1) % 20 == 0 or index == 0:
                progress = dict(checkpoint=label, images=index+1, total=len(loader), seconds=time.monotonic()-start)
                (root / 'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
            if limit and index+1 >= limit:
                break
    scores = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in metric.compute().items()}
    matched_records = [r for r in records if r['matched']]
    tp = len(matched_records)
    npred = sum(r['predicted_stems'] for r in image_records)
    summary = dict(checkpoint=str(checkpoint), checkpoint_label=label, device=device,
        dtype='bfloat16-autocast' if device=='cuda' else 'float32',
        images=len(image_records), confidence_threshold=0.3, matching_iou_threshold=0.5,
        stem_gt=len(records), stem_predictions=npred, stem_true_positives=tp,
        stem_recall_at_50=tp/max(1,len(records)), stem_precision_at_50=tp/max(1,npred),
        conditional_mean_cldice=float(np.mean([r['cldice'] for r in matched_records])) if tp else 0.,
        mean_cldice_with_misses_zero=sum(r['cldice'] for r in matched_records)/max(1,len(records)),
        conditional_mean_mask_iou=float(np.mean([r['matched_iou'] for r in matched_records])) if tp else 0.,
        fragmented_matched_count=sum(r['pred_components']>r['gt_components'] for r in matched_records),
        component_min_area_pixels=8, elapsed_seconds=time.monotonic()-start,
        metrics_full=scores)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'stem_instances.json').write_text(json.dumps(records, indent=2))
    (output / 'stem_predictions.json').write_text(json.dumps(image_records))
    print(json.dumps(summary), flush=True)
    del module, data, loader, metric
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = Path('/usr/local/rfdetr26/output/output1/baseline-seg-small-v1-seed42/training')
    for name, label in [('checkpoint_best_ema.pth','best_epoch63'),('last_ema.pth','last_epoch100')]:
        run(baseline/name, label, args.output, args.limit, args.device)
    (args.output/'completed.json').write_text(json.dumps({'completed':True}))
