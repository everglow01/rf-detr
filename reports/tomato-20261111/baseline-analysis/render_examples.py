"""Render selected validation crops from the already saved predictions; no inference."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

root = Path(__file__).parent
dataset = Path('/usr/local/rfdetr26/data/0/TOMATO.v7i.rfdetr-keypoints-v1/valid')
images = {r['id']: r for r in json.loads((dataset/'_annotations.coco.json').read_text())['images']}
labels = ['best_epoch63', 'last_epoch100']
predictions, instances = {}, {}
for label in labels:
    folder = root/'full'/label
    predictions[label] = {r['image_id']:r for r in json.loads((folder/'stem_predictions.json').read_text())}
    instances[label] = json.loads((folder/'stem_instances.json').read_text())

# Cases chosen from numerical diagnostics, not as a random estimate of error frequency.
selected = [(19, 5, 'High overlap'), (134, 12, 'High clDice, lower IoU'),
            (40, 10, 'Partial prediction'), (6, 11, 'Best misses at score >= 0.30'),
            (27, 10, 'Small isolated false region')]
fig, axes = plt.subplots(len(selected), 4, figsize=(12, 15), constrained_layout=True)
manifest = []
for row, (image_id, target_index, description) in enumerate(selected):
    image = np.asarray(Image.open(dataset/images[image_id]['file_name']).convert('RGB'))
    records = [r for r in instances[labels[0]] if r['image_id'] == image_id]
    gi = next(k for k, r in enumerate(records) if r['target_index'] == target_index)
    gt = mask_utils.decode(predictions[labels[0]][image_id]['gt_masks'][gi]).astype(bool)
    assert image.shape[:2] == gt.shape
    ys, xs = np.where(gt)
    x0, x1 = max(0, xs.min()-35), min(gt.shape[1], xs.max()+36)
    y0, y1 = max(0, ys.min()-35), min(gt.shape[0], ys.max()+36)
    if image_id == 27:
        # Include the remote 26-pixel island; the principal stem itself is continuous.
        x1 = max(x1, 360)
        y1 = max(y1, 145)
    crop = np.s_[y0:y1, x0:x1]
    for ax in axes[row]:
        ax.imshow(image[crop])
        ax.set_xticks([]); ax.set_yticks([])
    axes[row, 0].set_title(f'Image {image_id}: {description}', fontsize=9, wrap=True)
    axes[row, 1].set_title('Ground truth', fontsize=10)
    overlay = np.zeros((*gt.shape, 4))
    overlay[gt] = [0, 1, 1, .48]
    axes[row, 1].imshow(overlay[crop])
    for column, label in enumerate(labels, start=2):
        p = predictions[label][image_id]
        r = next(r for r in instances[label] if r['image_id']==image_id and r['target_index']==target_index)
        # All score-filtered stem predictions are displayed; the caption refers to the chosen GT.
        pm = np.zeros_like(gt)
        for encoded in p['pred_masks']:
            pm |= mask_utils.decode(encoded).astype(bool)
        overlay = np.zeros((*gt.shape, 4))
        overlay[pm] = [1, .35, 0, .48]
        axes[row, column].imshow(overlay[crop])
        axes[row, column].contour(gt[crop], levels=[.5], colors=['cyan'], linewidths=.8)
        axes[row, column].set_title(
            f"{'Best (63)' if column==2 else 'Last (100)'}: IoU {r['best_iou']:.3f}, clDice {r['cldice']:.3f}",
            fontsize=9)
        if image_id == 27 and column == 3:
            axes[row, column].add_patch(Circle((330-x0, 114.5-y0), 13, fill=False, color='yellow', lw=1))
            axes[row, column].annotate('26 px island', xy=(330-x0, 114.5-y0),
                                      xytext=(245-x0, 65-y0), color='yellow', fontsize=8,
                                      arrowprops={'arrowstyle':'->', 'color':'yellow'})
    manifest.append(dict(image_id=image_id, target_index=target_index, file_name=images[image_id]['file_name'],
                         crop_xyxy=[int(x0),int(y0),int(x1),int(y1)], selection_reason=description))
fig.suptitle('Stem segmentation examples | cyan: GT, orange: predictions (score >= 0.30)\n'
             'Selected diagnostic cases, not a representative random sample', fontsize=12)
fig.savefig(root/'stem_examples.png', dpi=150)
(root/'example_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Saved stem_examples.png and example_manifest.json')
