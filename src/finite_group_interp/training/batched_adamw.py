import torch
from torch import Tensor


class BatchedAdamW:
    """AdamW over a pytree of stacked params (leading ensemble dim N).

    Replicates torch.optim.AdamW's default (non-fused, non-amsgrad) update
    op-for-op so each ensemble slice equals an independent torch optimizer.
    """

    def __init__(
        self,
        params: dict[str, Tensor],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.exp_avg = {k: torch.zeros_like(v) for k, v in params.items()}
        self.exp_avg_sq = {k: torch.zeros_like(v) for k, v in params.items()}

    @torch.no_grad()
    def step(self, params: dict[str, Tensor], grads: dict[str, Tensor]) -> None:
        self.t += 1
        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t
        step_size = self.lr / bias_correction1
        bc2_sqrt = bias_correction2**0.5
        for k, p in params.items():
            g = grads[k]
            p.mul_(1 - self.lr * self.weight_decay)  # decoupled weight decay
            m, v = self.exp_avg[k], self.exp_avg_sq[k]
            m.lerp_(g, 1 - self.beta1)  # m = beta1*m + (1-beta1)*g
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            denom = (v.sqrt() / bc2_sqrt).add_(self.eps)
            p.addcdiv_(m, denom, value=-step_size)
