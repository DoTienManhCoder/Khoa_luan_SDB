"""Exponential Moving Average over model weights (full-model EMA).

Used by the EMA fine-tune study (`autoshotv2.train_ema`). This averages the
*entire* model state_dict, in contrast to the head-only EMA inside
`autoshotv2.train_phase2` (which only smooths the classification head trained on
cached features). See `reports/ema_study/` for the experiment.
"""

import copy

import torch


class ModelEMA:
    """Maintain a shadow copy of `model` whose weights are an EMA of training weights.

    Update rule (per call to `update`):
        theta_ema <- decay * theta_ema + (1 - decay) * theta_train

    Iterates the full state_dict so BatchNorm/running stats are also smoothed.
    Integer buffers (e.g. num_batches_tracked) are copied directly.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: torch.device | str | None = None) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.to(v.device, dtype=v.dtype), alpha=1.0 - self.decay)
            else:
                v.copy_(src.to(v.device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)
