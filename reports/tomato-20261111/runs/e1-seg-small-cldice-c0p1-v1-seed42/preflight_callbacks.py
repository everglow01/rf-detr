"""Check real two-GPU BF16 forwards and backwards before the full E1 run."""

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.distributed as dist


class ClDicePreflight(pl.Callback):
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.losses = []
        self.backward_steps = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        assert trainer.world_size == 2
        assert dist.get_backend() == 'nccl'
        assert pl_module.criterion.weight_dict['loss_mask_cldice'] == 0.1
        self.handle = pl_module.criterion.register_forward_hook(self.record_loss)
        torch.cuda.reset_peak_memory_stats()

    def record_loss(self, criterion, inputs, outputs) -> None:
        value = outputs['loss_mask_cldice'].detach()
        assert torch.isfinite(value).all(), 'Nonfinite clDice loss'
        self.losses.append({'training': bool(criterion.training), 'cldice': float(value)})

    def on_before_backward(self, trainer, pl_module, loss) -> None:
        assert torch.isfinite(loss).all(), 'Nonfinite total loss'

    def on_after_backward(self, trainer, pl_module) -> None:
        gradients = [p.grad for p in pl_module.parameters() if p.grad is not None]
        assert gradients, 'No gradients'
        assert all(bool(torch.isfinite(g).all()) for g in gradients), 'Nonfinite gradients'
        self.backward_steps += 1

    def on_fit_end(self, trainer, pl_module) -> None:
        self.handle.remove()
        assert self.backward_steps == 2, self.backward_steps
        assert len(self.losses) >= 4, self.losses
        assert any(r['cldice'] > 0 for r in self.losses), 'clDice never activated'
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if 'cldice' in k}
        assert metrics, 'clDice metric missing from Lightning logs'
        result = {'passed': True, 'rank': trainer.global_rank, 'world_size': trainer.world_size,
                  'backend': dist.get_backend(), 'backward_steps': self.backward_steps,
                  'criterion_calls': self.losses, 'logged_metrics': metrics,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir/f'preflight.rank{trainer.global_rank}.json').write_text(json.dumps(result, indent=2))
        print('E1_GPU_PREFLIGHT ' + json.dumps(result), flush=True)
