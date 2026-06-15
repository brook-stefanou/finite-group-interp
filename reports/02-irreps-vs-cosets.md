# Irreps vs cosets on a same-character-table pair: Dih(104) vs Dic(104)

> **Runs:** the order-104 sweep across 38 seeds, `runs/` (`pair-D52-*` and `pair-Dic26-*`, seeds 0–37) · **Figures + stats:** `uv run python scripts/pair_figures.py runs --wd wd1.0` · **Analysis code:** [`analysis/`](../src/finite_group_interp/analysis/)
>
> **Status: final at 38 seeds (weight decay 1.0).** The matrix-level and coset contrasts are taken on the **27 seeds where both groups grokked**; learnability is reported over all 38.

## Summary

Dih(104) and Dic(104) have **identical character tables** but different group structure — the dihedral group's 2-dimensional irreps are real, the dicyclic group's are quaternionic, and their subgroup/coset lattices differ. This is the smallest clean setting where character-level evidence runs out and the irreps-vs-cosets question has a real answer. Three findings across 38 seeds (27 matched seed pairs where both groups grokked):

1. **The pair separates on *learnability*, not on a matrix-level signature.** Dih groks at 35/38 seeds, fast (mean ~20k epochs) and never stuck below generalisation; Dic groks at 29/38, much later (mean ~40k), and at 6 seeds stays stuck in pure memorisation — a robust difficulty asymmetry tracking the quaternionic structure.
2. **The coset hypothesis gains no independent support over irreps.** Coset-membership decodability, measured against the model's *own* irreps as the control, has a mean *excess* of about −0.05 across all proper normal subgroups and seeds — zero or negative. The naive probe hits 100% — but so does the irrep control, which is exactly what exposes the probe as vacuous.
3. **The matrix-vs-trace R² gap — the instrument built to detect the real/quaternionic difference — does not separate the groups** (Welch p = 0.25 at n = 27). The larger sample confirms what n = 6 only suggested.

The headline: on this pair, **irreps are sufficient and cosets add nothing** — and the cleanest discriminator is how *hard* each group is to learn, not the post-hoc structure of the grokked weights. All three findings replicate on a fully-connected baseline (6 seeds/group), so they are not transformer artefacts — the coset-null in particular holds on the very architecture the coset account was originally read from.

## The question this pair can answer (and C₁₁₃ could not)

Two mechanistic accounts of group composition fit the published evidence: via the group's irreducible representations ([Chughtai et al., 2023](https://arxiv.org/abs/2302.03025)) and via coset/subgroup structure ([Stander et al., 2024](https://arxiv.org/abs/2312.06581)). Both bodies of evidence are character-level, and [report 01](01-c113-calibration.md) made the point that character-level evidence cannot separate them — C₁₁₃ is prime, so it has no subgroups and the coset account makes no prediction there.

Dih(104) and Dic(104) break that symmetry the right way. They share a character table, so any *character-level* instrument sees them as identical by construction; a calibrated instrument that nonetheless distinguishes them must be reading sub-character or subgroup-level structure — exactly the structure the two hypotheses disagree about. The dihedral group has a rich lattice of reflection subgroups; the dicyclic group has a unique involution and a quaternion-like lattice. If the model composes via cosets, the two should differ in coset decodability beyond what their (shared) irreps explain.

## Setup

- **Groups:** D52 = Dih(104) and Dic26 = Dic(104), both order 104, built from presentations (`scripts/run.py data.group=D52` / `Dic26`). They share a character table; their 2-dim irreps differ only in Frobenius–Schur type (real vs quaternionic).
- **Task / model:** identical to [report 01](01-c113-calibration.md) — predict a·b from (a, b) over all 104² pairs, 1-layer transformer, full-batch AdamW, deterministic CPU.
- **Sweep:** 38 seeds (0–37) × weight decay {0.5, 1.0} × 80k epochs, `stop_on_grok`. The matched comparison uses **weight decay 1.0 only** — Dic is grok-fragile at 0.5 (memorises within budget), so it is not a clean matched setting.
- **Matched seeds:** the **27** wd-1.0 seeds where *both* groups grokked. Comparing groups only at matched (seed, wd) is the only honest way to attribute a difference to the group rather than the hyperparameters.
- **Order-matched contrast:** C13⋊C8 (order 104, *different* character table, dim-4 irreps) was run as a control and **never groks at any setting** — a negative baseline returned to in [§ What this establishes](#what-this-establishes--and-what-it-cannot).

## What each hypothesis predicts

- **Irrep account:** the embedding concentrates in a few isotypic blocks (as calibrated on C₁₁₃); because the character tables are identical, this concentration looks the *same* for both groups at the character level. Any group difference must show in the *matrix-level* structure — the real-vs-quaternionic distinction the trace cannot see.
- **Coset account:** the model encodes coset membership for the group's subgroups, decodable from the residual stream *beyond* what the irreps explain, and differing between the two lattices.

The decisive measurements are therefore (a) the matrix-vs-trace **R² gap** (does the readout encode sub-character structure?) and (b) **coset decodability in excess of the irrep control** (is there a coset signal the irreps don't already provide?).

## Results

### Learnability: a robust, seed-stable asymmetry

![Grokking epoch by group, across seeds](../docs/figures/pair-grok-epochs.png)

This is the cleanest separator of the pair, and it does not require any post-hoc analysis of the weights.

- **Dih(104) groks at 35/38 seeds at wd 1.0**, fast: mean grok epoch **~20.1k** (± 11.8k, range ~4k–61k). The 3 non-grokking seeds are all *near-misses* (final test accuracy 0.935–0.985, just under the 0.99 grok threshold) — **none** are stuck in memorisation. It also groks at wd 0.5.
- **Dic(104) groks at 29/38 seeds, only at wd 1.0, and much later:** mean grok epoch **~39.7k** (± 16.4k, range ~12k–75k). Of the 9 that miss, **6 stay stuck in pure memorisation** (final test accuracy < 0.5, the lowest at 0.023, with test loss of 10–25 nats). Every wd-0.5 Dic run failed to grok.

The quaternionic group is consistently and substantially harder to learn, and the *kind* of failure differs: the dihedral group always reaches near-perfect generalisation, while the dicyclic group frequently never leaves memorisation. This is a sub-character property — the two groups are character-identical — surfacing as a difference in *optimisation*, not in the structure of the solution once found.

### The coset side adds nothing over irreps

![Coset excess over irrep control, by subgroup and seed](../docs/figures/pair-coset-excess.png)

For each proper normal subgroup H, a linear probe reads coset membership of a·b from the residual stream, scored against two controls: a random-partition null (capacity floor) and an **irrep-feature reference restricted to the model's kept irreps** (the non-vacuous control — by Peter–Weyl, *all* irreps would trivially reconstruct everything, so the reference uses only the blocks the model actually uses).

Across all 7 proper normal subgroups × 27 matched seeds (189 measurements per group), the **mean `excess_over_irrep` is −0.055 (Dih) and −0.044 (Dic)** — zero or negative. The pattern is consistent and instructive: on the subgroups where the naive probe hits **100%**, the irrep control *also* hits 100%, so the excess is ~0. The probe's apparent success is fully accounted for by the irreps the model already computes; there is no coset signal on top. This is the control both prior papers lack, and it is load-bearing: without it, the 100% probe accuracy reads as strong coset evidence; with it, that evidence evaporates.

**The coset side does not separate the pair, and shows no mechanism independent of the irreps.**

### The matrix-level R² gap

![Matrix-vs-trace R² gap, by group across seeds](../docs/figures/pair-fve-gap.png)

The functional-form fit regresses the logits onto the matrix elements of ρ(a)ρ(b)ρ(c)⁻¹ (the `full` fit) and onto the trace χ = tr ρ alone (the `trace` fit). The **gap = full − trace** is the variance explained by sub-character structure the character cannot see — and since Dih and Dic share a character table, the gap is precisely where a matrix-level difference between them would appear.

At n = 27 matched seeds the gap does not separate the groups:

```
D52 (Dih)   n=27  mean=0.074  std=0.064  range=[0.002, 0.217]
Dic26 (Dic) n=27  mean=0.055  std=0.054  range=[0.000, 0.163]
Welch t-test (Dih vs Dic): t=1.172  dof=50.7  p=0.2466  => NOT significant at alpha=0.05
```

The two distributions overlap heavily and the means are close (0.074 vs 0.055); the larger sample makes the non-separation clearer than the suggestive n = 6 result did, not weaker. What *is* robust at the matrix level: **both groups fill the full rank-4 in every 2-dimensional isotypic block** — so the naive "real uses rank 2, quaternionic uses rank 4" heuristic is dead for both.

### A caveat on the causal coset ablation

The coset analysis also reports a causal check — ablating the coset-direction subspace and measuring the increase in cross-coset error, against a matched random-partition subspace control ([commit `86856d5`](../src/finite_group_interp/analysis/coset_metrics.py)). That control matches the *capacity* of the ablated subspace but **not the irrep confound**: the coset subspace overlaps the irrep subspace the model needs, so the ablation excess is large and variable for *both* groups and does not separate them. The load-bearing coset result is therefore the **observational `excess_over_irrep`** above, not the ablation delta.

### The findings are not transformer artefacts

![Grokking epoch by group on the fully-connected baseline](../docs/figures/fc-grok-epochs.png)

The coset hypothesis was originally read off fully-connected networks, so a transformer-only result leaves architecture as a stated confound. Re-running the pair on a one-hidden-layer FC network (shared embedding of a and b, concatenated into a single ReLU layer, no biases; 6 seeds per group at wd 1.0) reproduces all three findings:

- **Learnability asymmetry holds.** Dih groks at **6/6** seeds, fast (mean ~5.6k epochs, range 4.1k–6.7k); Dic groks at **6/6**, ~3× later (mean ~16.1k, range 14.7k–17.3k). The ordering — quaternionic harder than real — is identical to the transformer's. One difference: on the FC every Dic seed *does* eventually grok, so the catastrophic memorisation-plateau failure mode seen on the transformer (6/38 stuck) appears to be transformer-specific; on the FC the asymmetry is purely a difference in *speed*.
- **Coset-null holds — on the very architecture that generated the coset hypothesis.** Mean `excess_over_irrep` is **+0.05 (Dih)** and **−0.04 (Dic)** across all 7 proper normal subgroups × 6 seeds; both centre on zero. The Dih spread is noisier than on the transformer (std 0.15, one normal subgroup reaching +0.55) rather than cleanly negative, but Dic is negative and neither group shows a systematic coset signal beyond its own irreps.
- **Matrix-level R² gap stays null.** Dih mean 0.021, Dic mean 0.013 (Welch t = 1.51, dof = 6.5, **p = 0.18**) — the gaps are if anything *smaller* than on the transformer, and still do not separate the groups.

So the learnability asymmetry is architecture-general, and cosets add nothing over irreps even on the fully-connected architecture that the coset account was built from.

## What this establishes — and what it cannot

Established (38 seeds; matrix-level and coset at the 27 matched seeds):

1. **A learnability asymmetry** tied to the quaternionic structure: the character-identical dicyclic group is reliably harder to grok than the dihedral group, and fails in a qualitatively different way (memorisation plateaus vs near-misses). Difficulty appears to scale with the representation-theoretic complexity the character table hides.
2. **No coset mechanism beyond the irreps** on this pair, under the irrep-restricted control that prior coset evidence omits.
3. **No matrix-level signature** of the real/quaternionic difference at the level the R² gap measures — the natural instrument for it returns a null (p = 0.25). This is itself a finding: the readout does not appear to encode the distinction the gap was built to detect.
4. **Architecture-generality of (1) and (2).** The learnability asymmetry and the coset-null both replicate on a fully-connected baseline (6 seeds/group) — including the coset-null *on the architecture the coset account was originally read from*. The findings are not transformer artefacts.

Not established:

- **Generality beyond dim-2 irreps.** The whole pair lives in 2-dimensional blocks, where the gap has the least room. The order-125 secondary pair (Heisenberg/F₅ vs C₂₅⋊C₅) has **dimension-5** irreps and is the natural next discriminator. A go/no-go run on Heisenberg/F₅ did **not** grok — at the base setting (1M epochs) or across a small weight-decay × train-fraction probe (best test accuracy ~0.15) — so the dim-5 regime is, for now, out of reach at this data budget rather than a place the gap has been made to speak. This is consistent with the difficulty ladder below (dim-1 easy → dim-2 harder → dim-4/5 not learned), but it means the dim-5 matrix contrast remains open.

The order-matched negative is worth stating plainly: C13⋊C8 (different character table, dim-4 irreps) never grokked in 80k epochs at any setting. Read alongside "Dih easy, Dic harder," this hints at a difficulty ladder that climbs with representation-theoretic complexity — a thread the order-125 pair will test directly.

## Next

- **Secondary pair, dim-5:** Heisenberg/F₅ vs C₂₅⋊C₅ — same character table, far richer matrix structure, where the R² gap would have the most room to either speak or be conclusively silent. The Heisenberg go/no-go did not grok at the current data budget (see above), so reaching this regime needs either more data (larger train fraction) or an architectural/optimisation change before the matrix contrast can be run.

## Reproduce

```bash
uv sync
# train the pair across seeds (parallel local sweep; env-overridable):
GROUPS=Dic26,D52 SEEDS=0-37 uv run python scripts/sweep_parallel.py
# cross-seed figures + stats (learnability, matrix-level R² gap, coset excess):
uv run python scripts/pair_figures.py runs --wd wd1.0 --out docs/figures
# fully-connected baseline (architecture confound): same pair, FC architecture
GROUPS=Dic26,D52 SEEDS=0-5 ARCH=fc uv run python scripts/sweep_parallel.py
uv run python scripts/pair_figures.py runs/<date> --arch fc --out docs/figures
```
