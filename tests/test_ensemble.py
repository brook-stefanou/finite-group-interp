import numpy as np
import torch
from torch.testing import assert_close


def test_seed_batches_match_train_test_split():
    from finite_group_interp.groups.catalog import resolve_group
    from finite_group_interp.task import build_group_task, train_test_split
    from finite_group_interp.training.ensemble import build_seed_batches

    group = resolve_group("S3")  # small, order 6 -> O^2 = 36
    seeds = [7, 11]
    batches = build_seed_batches(group, train_frac=0.4, seeds=seeds, device="cpu")

    task = build_group_task(group)
    for i, seed in enumerate(seeds):
        split = train_test_split(task, 0.4, seed)
        # tokens are (a, b, '='); '=' id == group.order
        exp_train = np.concatenate(
            [split.train_inputs, np.full((len(split.train_inputs), 1), group.order)], axis=1
        )
        assert_close(batches.train_tokens[i], torch.tensor(exp_train, dtype=torch.long))
        assert_close(batches.train_targets[i], torch.tensor(split.train_targets, dtype=torch.long))


def test_stacked_init_matches_single_build_model():
    from finite_group_interp.groups.catalog import resolve_group
    from finite_group_interp.training.config import GrokkingConfig
    from finite_group_interp.training.trainer import build_model, set_seed
    from finite_group_interp.training.ensemble import stack_seeded_models

    cfg = GrokkingConfig(experiment={"name": "x", "seed": 0}, data={"group": "S3"})
    group = resolve_group("S3")
    seeds = [3, 9]
    _, params, _ = stack_seeded_models(cfg, group, seeds, device="cpu")

    for i, seed in enumerate(seeds):
        set_seed(seed)
        ref = build_model(cfg, group)
        ref_sd = dict(ref.named_parameters())
        for name, stacked in params.items():
            assert_close(stacked[i], ref_sd[name])
