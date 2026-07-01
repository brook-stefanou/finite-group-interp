import torch
from torch.testing import assert_close


def test_batched_adamw_matches_torch_per_slice():
    from same_character_table_interp.training.batched_adamw import BatchedAdamW

    torch.manual_seed(0)
    N, shape = 3, (4, 5)
    init = torch.randn(N, *shape, dtype=torch.float64)

    # Reference: N independent torch AdamW optimizers on cloned slices.
    ref_params = [init[i].clone().requires_grad_(True) for i in range(N)]
    ref_opts = [
        torch.optim.AdamW([ref_params[i]], lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0)
        for i in range(N)
    ]

    # Batched: one optimizer over the stacked [N, *shape] param.
    batched = {"w": init.clone()}
    opt = BatchedAdamW(batched, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0)

    torch.manual_seed(1)
    for _ in range(10):
        grads_per = [torch.randn(*shape, dtype=torch.float64) for _ in range(N)]
        # reference step
        for i in range(N):
            ref_opts[i].zero_grad()
            ref_params[i].grad = grads_per[i].clone()
            ref_opts[i].step()
        # batched step
        g = torch.stack(grads_per, dim=0)
        opt.step(batched, {"w": g})

    ref_stacked = torch.stack([p.detach() for p in ref_params], dim=0)
    assert_close(batched["w"], ref_stacked, rtol=1e-10, atol=1e-12)
