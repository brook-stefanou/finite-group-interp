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


def test_batched_first_step_matches_single_run():
    import torch.nn.functional as F
    from finite_group_interp.groups.catalog import resolve_group
    from finite_group_interp.training.config import GrokkingConfig
    from finite_group_interp.training.trainer import build_model, set_seed
    from finite_group_interp.training.ensemble import (
        build_seed_batches,
        stack_seeded_models,
        make_grad_fn,
    )

    cfg = GrokkingConfig(experiment={"name": "x", "seed": 0}, data={"group": "S3"})
    group = resolve_group("S3")
    seed = 5

    # single-run reference: loss + grad on the seed's train split
    set_seed(seed)
    model = build_model(cfg, group)
    batches = build_seed_batches(group, 0.4, [seed], device="cpu")
    logits = model(batches.train_tokens[0])
    ref_loss = F.cross_entropy(logits[:, -1, :], batches.train_targets[0])
    ref_grads = torch.autograd.grad(ref_loss, [model.W_E])[0]

    # batched (N=1)
    base, params, buffers = stack_seeded_models(cfg, group, [seed], device="cpu")
    grad_fn = make_grad_fn(base)
    grads, losses = grad_fn(params, buffers, batches.train_tokens, batches.train_targets)
    assert_close(losses[0], ref_loss)
    assert_close(grads["W_E"][0], ref_grads)


def test_member_writer_output_loads_via_analysis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # run dirs are created under ./runs
    from finite_group_interp.groups.catalog import resolve_group
    from finite_group_interp.training.config import GrokkingConfig
    from finite_group_interp.training.ensemble import (
        MemberWriter,
        slice_state_dict,
        stack_seeded_models,
    )
    from finite_group_interp.analysis.loading import load_run  # see Step 3 note

    cfg = GrokkingConfig(experiment={"name": "x", "seed": 0}, data={"group": "S3"})
    group = resolve_group("S3")
    base, params, buffers = stack_seeded_models(cfg, group, [5], device="cpu")

    writer = MemberWriter(cfg, seed=5)
    sd = slice_state_dict(params, buffers, 0)
    writer.save_checkpoint("step_0", sd, epoch=0)
    writer.write_metrics(
        [
            {
                "step": 0,
                "train_loss": 1.0,
                "train_acc": 0.0,
                "test_loss": 1.0,
                "test_acc": 0.0,
                "weight_norm": 1.0,
            },
        ]
    )
    writer.finalize({"test_acc": 0.0})

    run = load_run(writer.run_dir)
    assert run.metrics[0]["step"] == 0
    assert (writer.run_dir / "checkpoints" / "step_0.pt").exists()
    assert (writer.run_dir / "manifest.json").exists()


def test_run_ensemble_produces_one_dir_per_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from finite_group_interp.training.config import GrokkingConfig
    from finite_group_interp.training.ensemble import run_ensemble

    cfg = GrokkingConfig(
        experiment={"name": "ens", "seed": 0, "use_wandb": False},
        data={"group": "S3", "train_frac": 0.5},
        optim={"epochs": 50, "log_every": 10, "stop_on_grok": True},
        ensemble={"enabled": True, "seeds": [1, 2]},
    )
    dirs = run_ensemble(cfg)
    assert len(dirs) == 2
    for d in dirs:
        assert (d / "manifest.json").exists()
        assert (d / "metrics.jsonl").exists()
