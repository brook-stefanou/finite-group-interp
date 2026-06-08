"""Coset-side metrics for the irreps-vs-cosets discriminator.

Does a trained model encode and causally use coset structure relative to the
group's subgroups? A linear probe on the '='-position residual decodes coset
membership; a random-partition null and an irrep-feature control keep the claim
honest; a coset-direction ablation tests causal use. Calibrated on planted
ground truth (the only known answer until the pairs are trained).
"""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch

from finite_group_interp.analysis.cache import forward_with_cache
from finite_group_interp.groups.group import Element, FiniteGroup
from finite_group_interp.model import OneLayerTransformer
from finite_group_interp.representations.irreps import extract_irreps
from finite_group_interp.task import build_group_task

Target = Literal["a", "b", "ab"]


@dataclass(frozen=True)
class CosetResult:
    subgroup_size: int  # |H|
    n_cosets: int  # k = |G| / |H|
    target: Target  # which element's coset was probed
    probe_acc: float  # held-out coset accuracy on model activations
    random_null_mean: float  # mean held-out acc over shuffled-label probes
    random_null_std: float
    irrep_ref_acc: float  # same probe on the model's-irreps features (nan if skipped)
    excess_over_null: float  # probe_acc - random_null_mean
    excess_over_irrep: float  # probe_acc - irrep_ref_acc (nan if skipped)
    ablation_delta_loss: float  # coset-subspace ablation, total CE-loss increase
    random_ablation_delta_loss: float  # same-rank random-subspace control
    ablation_cross_coset_delta: float  # increase in cross-coset error rate
    ablation_within_coset_delta: float  # increase in within-coset error rate


def _target_elements(group: FiniteGroup, target: Target) -> list[Element]:
    """The target element (a, b, or ab) for every (a,b) example, row-major."""
    task = build_group_task(group)
    el = group.elements
    if target == "a":
        return [el[int(i)] for i in task.inputs[:, 0]]
    if target == "b":
        return [el[int(i)] for i in task.inputs[:, 1]]
    return [el[int(i)] for i in task.targets]  # ab


def _coset_of_map(group: FiniteGroup, H: list[Element]) -> dict[Element, int]:
    """{element: coset index} for the left cosets of H."""
    return {e: ci for ci, c in enumerate(group.left_cosets(H)) for e in c}


def coset_labels(group: FiniteGroup, H: list[Element], target: Target) -> np.ndarray:
    """Coset-of-{a|b|ab} index for every (a,b) example. Shape [|G|^2]."""
    coset_of = _coset_of_map(group, H)
    return np.array([coset_of[e] for e in _target_elements(group, target)])


def _all_pairs_resid(
    model: OneLayerTransformer, group: FiniteGroup
) -> tuple[np.ndarray, torch.Tensor]:
    """('='-position residual [|G|^2, d_model], targets [|G|^2]) over all pairs."""
    n = group.order
    task = build_group_task(group)
    eq = np.full((task.inputs.shape[0], 1), n)
    tokens = torch.tensor(np.concatenate([task.inputs, eq], axis=1), dtype=torch.long)
    cache = forward_with_cache(model, tokens)
    resid: np.ndarray = cache["resid_final"][:, -1, :].detach().numpy()
    targets = torch.tensor(task.targets, dtype=torch.long)
    return resid, targets


def fit_linear_probe(
    x: np.ndarray, y: np.ndarray, *, seed: int, l2: float = 1e-3, max_iter: int = 100
) -> float:
    """Held-out accuracy of a linear (logistic-regression) probe.

    A torch nn.Linear + cross-entropy fit with LBFGS (convex 2nd-order solver,
    deterministic). Features z-scored on the train split; small L2 keeps the
    solve well-posed. 80/20 seeded split; returns test accuracy.
    """
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n, d = xt.shape
    k = int(yt.max().item()) + 1
    gen = torch.Generator().manual_seed(seed)
    tr_parts, te_parts = [], []
    for c in range(k):
        idx_c = torch.where(yt == c)[0]
        idx_c = idx_c[torch.randperm(len(idx_c), generator=gen)]
        n_tr = max(1, int(0.8 * len(idx_c)))
        tr_parts.append(idx_c[:n_tr])
        te_parts.append(idx_c[n_tr:])
    tr = torch.cat(tr_parts)
    te = torch.cat(te_parts)
    if len(te) == 0:
        return float("nan")
    mu = xt[tr].mean(0, keepdim=True)
    sd = xt[tr].std(0, keepdim=True) + 1e-6
    xs = (xt - mu) / sd
    probe = torch.nn.Linear(d, k)
    with torch.no_grad():
        probe.weight.normal_(generator=gen)
        probe.bias.zero_()
    opt = torch.optim.LBFGS(probe.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")
    x_tr, y_tr = xs[tr], yt[tr]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(x_tr), y_tr)
        reg = torch.stack([(p**2).sum() for p in probe.parameters()]).sum()
        loss = loss + l2 * reg
        torch.autograd.backward(loss)
        return loss

    # torch's LBFGS.step(closure) is untyped in the stubs; suppress inline so the
    # module is self-contained (no separate pyproject override that a partial
    # commit would leave un-applied).
    opt.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        acc = (probe(xs[te]).argmax(-1) == yt[te]).float().mean().item()
    return float(acc)


def random_partition_null(
    x: np.ndarray, y: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float]:
    """Mean/std held-out probe accuracy over label *shuffles* (matched class
    sizes, no relation to x) -- the probe-capacity floor."""
    rng = np.random.default_rng(seed)
    accs = [fit_linear_probe(x, rng.permutation(y), seed=seed * 1000 + d + 1) for d in range(draws)]
    return float(np.mean(accs)), float(np.std(accs))


def _irrep_features(
    group: FiniteGroup, keep_irreps: list[int], target_elements: list[Element]
) -> np.ndarray:
    """[N, sum 2*d_i^2] real features: real+imag matrix elements of the kept
    irreps, evaluated at each example's target element."""
    irreps = extract_irreps(group)
    cols: list[np.ndarray] = []
    for j in keep_irreps:
        mats = np.stack([irreps[j].matrices[e] for e in target_elements], axis=0)
        flat = mats.reshape(len(target_elements), -1)
        cols.append(flat.real)
        cols.append(flat.imag)
    return np.concatenate(cols, axis=1)


def coset_probe_suite(
    resid: np.ndarray,
    group: FiniteGroup,
    H: list[Element],
    target: Target,
    keep_irreps: list[int],
    *,
    seed: int,
    null_draws: int = 5,
) -> tuple[float, float, float, float, int]:
    """(probe_acc, null_mean, null_std, irrep_ref_acc, n_cosets). irrep_ref_acc
    is nan when keep_irreps is empty."""
    y = coset_labels(group, H, target)
    k = len(group.left_cosets(H))
    probe_acc = fit_linear_probe(resid, y, seed=seed)
    null_mean, null_std = random_partition_null(resid, y, draws=null_draws, seed=seed)
    if keep_irreps:
        x_irr = _irrep_features(group, keep_irreps, _target_elements(group, target))
        irrep_ref = fit_linear_probe(x_irr, y, seed=seed)
    else:
        irrep_ref = float("nan")
    return probe_acc, null_mean, null_std, irrep_ref, k


def coset_subspace(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthonormal basis (rows, rank <= k-1) of the span of the per-coset mean
    offsets (class_mean_c - global_mean) -- the subspace that linearly separates
    cosets."""
    gmean = x.mean(0)
    diffs = np.stack([x[y == c].mean(0) - gmean for c in np.unique(y)], axis=0)
    u, s, vh = np.linalg.svd(diffs, full_matrices=False)
    rank = int(np.sum(s > 1e-8 * max(s[0], 1.0)))
    return vh[:rank]


def _random_subspace(rank: int, d: int, seed: int) -> np.ndarray:
    """[rank, d] orthonormal rows, a random subspace of the given rank.

    The caller must pass a seed INDEPENDENT of any used to generate the
    data/signal directions -- reusing such a seed replays the same draws and
    the "random" subspace can align with the signal.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    return cast(np.ndarray, q.T)


def _ce_and_cross(
    resid: np.ndarray, w_u: np.ndarray, targets: torch.Tensor, n: int, coset_of: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross-entropy at the answer; the argmax prediction; and a bool array
    'argmax is in a different coset than the true answer'."""
    logits = torch.tensor(resid @ w_u, dtype=torch.float32)[:, :n]
    ce = torch.nn.functional.cross_entropy(logits, targets).item()
    pred = logits.argmax(-1).numpy()
    cross = coset_of[pred] != coset_of[targets.numpy()]
    return float(ce), pred, cross


def ablate_coset_direction(
    model: OneLayerTransformer,
    group: FiniteGroup,
    H: list[Element],
    target: Target,
    resid: np.ndarray,
    targets: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    """Project the coset subspace out of resid, recompute logits = resid @ W_U,
    measure damage vs a matched random-subspace control, split into cross-coset
    / within-coset error-rate increase. Non-destructive."""
    n = group.order
    w_u = model.W_U.detach().numpy()
    y = coset_labels(group, H, target)
    el_coset = _coset_of_map(group, H)
    coset_of = np.array([el_coset[group.elements[i]] for i in range(n)])

    b = coset_subspace(resid, y)
    r_abl = resid - resid @ (b.T @ b)
    q = _random_subspace(b.shape[0], resid.shape[1], seed)
    r_rand = resid - resid @ (q.T @ q)

    tgt = targets.numpy()
    base_ce, pred_base, base_cross = _ce_and_cross(resid, w_u, targets, n, coset_of)
    abl_ce, pred_abl, abl_cross = _ce_and_cross(r_abl, w_u, targets, n, coset_of)
    rand_ce, _, _ = _ce_and_cross(r_rand, w_u, targets, n, coset_of)

    base_within = float(((pred_base != tgt) & ~base_cross).mean())
    abl_within = float(((pred_abl != tgt) & ~abl_cross).mean())

    return {
        "ablation_delta_loss": abl_ce - base_ce,
        "random_ablation_delta_loss": rand_ce - base_ce,
        "ablation_cross_coset_delta": float(abl_cross.mean()) - float(base_cross.mean()),
        "ablation_within_coset_delta": abl_within - base_within,
    }


def coset_analysis(
    model: OneLayerTransformer,
    group: FiniteGroup,
    keep_irreps: list[int],
    *,
    seed: int = 0,
) -> list[CosetResult]:
    """Run the coset instrument over every proper nontrivial subgroup x target.

    ``keep_irreps``: character-table rows of the irreps the model uses (energy-
    kept set) for the irrep-feature control; pass [] to skip it. Returns [] for
    groups with no proper subgroups (e.g. prime cyclic) -- the forward pass is
    skipped in that case.
    """
    proper = [h for h in group.subgroups() if 1 < len(h) < group.order]
    if not proper:
        return []
    resid, targets = _all_pairs_resid(model, group)
    targets_list: list[Target] = ["a", "b", "ab"]
    results: list[CosetResult] = []
    for h in proper:
        for tgt in targets_list:
            pa, nm, ns, ir, k = coset_probe_suite(resid, group, h, tgt, keep_irreps, seed=seed)
            abl = ablate_coset_direction(model, group, h, tgt, resid, targets, seed=seed)
            results.append(
                CosetResult(
                    subgroup_size=len(h),
                    n_cosets=k,
                    target=tgt,
                    probe_acc=pa,
                    random_null_mean=nm,
                    random_null_std=ns,
                    irrep_ref_acc=ir,
                    excess_over_null=pa - nm,
                    excess_over_irrep=pa - ir,
                    ablation_delta_loss=abl["ablation_delta_loss"],
                    random_ablation_delta_loss=abl["random_ablation_delta_loss"],
                    ablation_cross_coset_delta=abl["ablation_cross_coset_delta"],
                    ablation_within_coset_delta=abl["ablation_within_coset_delta"],
                )
            )
    return results
