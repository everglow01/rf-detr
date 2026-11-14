#!/usr/bin/env bash
# Run the user-authorized beta=1 and beta=2 full experiments sequentially.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
expected_commit="${1:?Pass the verified full Git commit SHA}"
output="$repo/output/output1/cldice-beta100-v1-seed42"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 RFDETR_SKIP_DINOV2_PREWARM=1
export PYTHONPATH="$repo/src:$repo/reports/tomato-20261111/runs/e1-seg-small-cldice-c0p1-v1-seed42"

test "$repo" = /usr/local/rfdetr26
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain --untracked-files=no)"
exec 9> "$repo/output/output1/tomato-experiment.lock"
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test ! -e "$output"
mkdir -- "$output"
trap 'code=$?; printf "\nPIPELINE_END exit=%s utc=%s\n" "$code" "$(date -u +%FT%TZ)"' EXIT

.venv/bin/python -B - "$output" "$expected_commit" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
repo = Path.cwd()
configs = []
paths = [repo / 'scripts/train_tomato_beta100.sh',
         repo / 'reports/tomato-20261111/runs/e1-seg-small-cldice-c0p1-v1-seed42/benchmark_callbacks.py',
         repo / 'checkpoint/pretrained/rf-detr-seg-small.pt']
paths.extend(repo / 'data/0/TOMATO.v7i.rfdetr-keypoints-v1' / split / '_annotations.coco.json'
             for split in ('train', 'valid'))
for arm, beta in (('beta1', 1.0), ('beta2', 2.0)):
    path = repo / f'configs/tomato/seg_small_cldice_{arm}_100.yaml'
    paths.append(path)
    config = yaml.safe_load(path.read_text())
    train = config['model']['train_config']['init_args']
    assert train['epochs'] == 100 and not train['early_stopping']
    assert train['mask_cldice_recall_beta'] == beta and train['mask_cldice_loss_coef'] == 0.1
    assert (train['mask_cldice_iterations'], train['mask_cldice_chunk_size']) == (25, 4)
    assert train['output_dir'] == str(root / arm)
    assert config['trainer']['callbacks'] == [dict(
        class_path='benchmark_callbacks.BenchmarkTiming', init_args=dict(output_dir=str(root / arm)))]
    (root / f'{arm}.yaml').write_bytes(path.read_bytes())
    train['mask_cldice_recall_beta'] = 1.0
    train['output_dir'] = 'ARM'
    config['trainer']['callbacks'][0]['init_args']['output_dir'] = 'ARM'
    configs.append(config)
assert configs[0] == configs[1]
metadata = dict(git_commit=sys.argv[2], epochs_per_arm=100, order=['beta1', 'beta2'],
                world_size=2, per_gpu_batch=8, initialization='official pretrained; no resume',
                analysis='deferred until user requests after completion',
                sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with (root / 'provenance.json').open('x') as stream:
    json.dump(metadata, stream, indent=2)
print('PREFLIGHT_OK: only beta and output paths differ; both arms require 100 epochs.', flush=True)
PY

git archive -o "$output/source.tar.gz" HEAD src scripts configs/tomato \
  reports/tomato-20261111/runs/e1-seg-small-cldice-c0p1-v1-seed42/benchmark_callbacks.py

for arm in beta1 beta2; do
  test "$(git rev-parse HEAD)" = "$expected_commit"
  test -z "$(git status --porcelain --untracked-files=no)"
  mkdir -- "$output/$arm"
  printf '\nTRAIN_START arm=%s epochs=100 utc=%s\n' "$arm" "$(date -u +%FT%TZ)"
  # Convert carriage-return progress updates to flushed lines visible in the GPU task panel.
  .venv/bin/python -B -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    --module rfdetr fit --config "$output/$arm.yaml" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | tee "$output/$arm/train.log"
  .venv/bin/python -B - "$output/$arm" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
for rank in (0, 1):
    timing = json.loads((output / f'timing.rank{rank}.json').read_text())
    assert timing['status'] == 'completed'
    assert len(timing['epoch_records']) == 100
    assert all(row['complete'] for row in timing['epoch_records'])
with (output / 'completed.json').open('x') as stream:
    json.dump(dict(complete=True, epochs=100, ranks=2), stream, indent=2)
print(f'TRAIN_COMPLETE arm={output.name} epochs=100', flush=True)
PY
done
printf '{"complete":true,"epochs_per_arm":100,"arms":["beta1","beta2"]}\n' > "$output/completed.json"
printf '\nALL_TRAINING_COMPLETE: beta1 and beta2 each finished 100 epochs.\n'
