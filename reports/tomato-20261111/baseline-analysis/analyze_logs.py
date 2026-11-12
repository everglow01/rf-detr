"""Summarize epoch-aligned training evidence and export reproducible figures."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent
rows = list(csv.DictReader((root/'raw/metrics.csv').open(encoding='utf-8')))
epochs = {}
for row in rows:
    if row.get('epoch'):
        epoch = int(row['epoch']) + 1
        epochs.setdefault(epoch, {'epoch': epoch})
        epochs[epoch].update({k: float(v) for k,v in row.items() if v and k not in ('epoch','step')})
series = [epochs[e] for e in sorted(epochs)]
assert len(series) == 100
key = 'val/ema_segm_mAP_50_95'
best = max(series, key=lambda r:r[key])
lowest_loss = min(series, key=lambda r:r['val/loss'])
periods = []
for lo,hi in [(1,5),(6,20),(21,40),(41,60),(61,70),(71,80),(81,100)]:
    values = [r for r in series if lo<=r['epoch']<=hi]
    periods.append({'epochs':[lo,hi], **{k:statistics.mean(r[k] for r in values) for k in
                    [key,'val/ema_segm_mAP_50','val/ema_AP/stem','train/loss','val/loss',
                     'train/loss_mask_dice','val/loss_mask_dice','val/loss_ce']}})
timing = json.loads((root/'raw/timing.rank0.json').read_text())
summary = {'best_overall_mask_epoch':best['epoch'], 'best_overall_mask_ap':best[key],
           'lowest_validation_loss_epoch':lowest_loss['epoch'], 'lowest_validation_loss':lowest_loss['val/loss'],
           'last_overall_mask_ap':series[-1][key], 'drop_from_best_percentage_points':100*(best[key]-series[-1][key]),
           'fit_seconds':timing['fit_elapsed_seconds'],
           'median_epoch_seconds':statistics.median(r['epoch_total_seconds'] for r in timing['epoch_records'][1:]),
           'selected_epochs':[epochs[e] for e in [5,10,20,22,40,60,63,70,80,100]], 'periods':periods,
           'epoch_numbering':'1-based; metrics.csv and best checkpoint store zero-based epoch',
           'per_class_logged_ap_type':'bbox, not segmentation'}
(root/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (root/'epoch_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(series[0]))
    writer.writeheader(); writer.writerows(series)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
x=[r['epoch'] for r in series]
axes[0,0].plot(x,[r[key]*100 for r in series],label='Mask AP50:95',color='#126c9c')
axes[0,0].scatter([best['epoch']],[best[key]*100],color='#bb4a3b',zorder=5)
axes[0,0].annotate(f"Best: epoch {best['epoch']}, {best[key]*100:.2f}%",(best['epoch'],best[key]*100),
                   xytext=(22,62),arrowprops={'arrowstyle':'->','color':'#555'})
axes[0,0].set(title='Validation: overall segmentation',ylabel='AP (%)',ylim=(52,71))
axes[0,1].plot(x,[r['train/loss'] for r in series],label='Train (augmented, live weights)',color='#126c9c')
axes[0,1].plot(x,[r['val/loss'] for r in series],label='Validation (EMA)',color='#bb4a3b')
axes[0,1].set(title='Loss trends (different evaluation conditions)',ylabel='Total loss')
axes[0,1].legend(fontsize=8)
for name,color in [('ripe','#af4835'),('stem','#33865c'),('unripe','#a0852a')]:
    axes[1,0].plot(x,[r[f'val/ema_AP/{name}']*100 for r in series],label=name,color=color)
axes[1,0].set(title='Logged per-class AP is BOX AP',ylabel='Box AP50:95 (%)')
axes[1,0].legend()
axes[1,1].plot(x,[r['train/loss_mask_dice'] for r in series],label='Train mask Dice loss',color='#126c9c')
axes[1,1].plot(x,[r['val/loss_mask_dice'] for r in series],label='Validation mask Dice loss',color='#bb4a3b')
axes[1,1].set(title='Final decoder mask Dice loss',ylabel='Unweighted loss')
axes[1,1].legend(fontsize=8)
for ax in axes.flat:
    ax.axvline(70.5,ls='--',color='#777',lw=1,label='LR /10 from epoch 71')
    ax.set_xlabel('Epoch (1-based)')
    ax.grid(alpha=.15)
fig.suptitle('RF-DETR-Seg Small baseline | 100 epochs | seed 42 | dashed line: LR drop',fontsize=13)
fig.savefig(root/'baseline_curves.png',dpi=180)
fig.savefig(root/'baseline_curves.pdf')
print(json.dumps({k:v for k,v in summary.items() if k not in ('selected_epochs','periods')},indent=2))
