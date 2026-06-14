# finite-group-interp

![CI](https://github.com/stefanoubrook/finite-group-interp/actions/workflows/ci.yml/badge.svg)

**When a neural network learns a finite group operation, what algorithm is it actually implementing internally?**

A mechanistic interpretability research codebase: reproducible grokking experiments on finite-group multiplication, plus the representation-theory machinery needed to reverse-engineer what the trained networks compute.

---

## Repository Structure

```
finite-group-interp/
├── src/finite_group_interp/
│   ├── groups/             # finite group algebra: construction, presentations, catalog
│   ├── representations/    # character tables, isotypic projectors
│   ├── model.py            # 1-layer transformer (from scratch, analysis-friendly)
│   ├── task.py             # group-multiplication task + train/test split
│   ├── training/           # trainer, config schemas, run manifests, logging
│   └── analysis/           # checkpoint loading + activation cache; irrep/coset metrics in progress
├── scripts/                # run.py (training), analyze_run.py (irrep analysis)
├── reports/                # committed research write-ups
├── tests/                  # 180+ tests incl. mathematical property tests
└── runs/                   # local run artifacts (gitignored)
```

How data flows through the pipeline — every artifact traceable back to a config and a commit:

```
config (CLI dotlist → validated Pydantic schema)
   │
   ▼
scripts/run.py ── trainer ──► runs/<date>/<run_id>/
                                ├── manifest.json      provenance: git hash, config hash, env
                                ├── metrics.jsonl      training curves, every eval
                                └── checkpoints/*.pt   event-dense weight snapshots,
                                       │               full config embedded in each
                                       ▼
scripts/analyze_run.py ──────► runs/<run_id>/analysis/
                                ├── metrics.json       every reported number + provenance
                                └── figures/*.png ───► docs/figures/ ───► reports/*.md
                                                       (published)       (hand-written around
                                                                          generated artifacts)
```

---

## The Research Question

When a network learns to compose elements of a finite group, *what algorithm and internal representation does it use* — and how does that depend on the group's structure? This engages an open debate in the literature: do networks compose via the group's **irreducible representations** ([Chughtai et al., 2023](https://arxiv.org/abs/2302.03025)) or via **coset / subgroup structure** ([Stander et al., 2024](https://arxiv.org/abs/2312.06581))?

The published evidence on both sides is **character-level** (correlations between model outputs/weights and irreducible characters) — and both hypotheses fit that evidence. Neither paper varied the *group* to a regime where the hypotheses make divergent predictions. That is the gap this project targets.

### Findings so far

* **Groups of order < 20 do not generalise.** Across S₃, Q₈, A₄, C₈ × weight-decay sweeps (150k epochs), every model memorised the training set quickly and stayed at chance test accuracy. The dataset is the bottleneck (|G|² ≤ 361 examples), not optimisation — so small groups are learning a lookup table, not the group operation, and any "algorithm" read off them would be an artefact. The investigation therefore runs on groups of order ≈ 100–350.
* **Pipeline validated at scale.** C₁₁₃ (modular addition, the canonical grokking task) groks cleanly: train_frac 0.3, 30k epochs → test accuracy **99.98%**, with the weight-norm progress measure and dense checkpoints captured through the transition.
* **C₁₁₃ is calibration, not evidence in the debate — by construction.** 113 is prime, so C₁₁₃ has no proper subgroups: the coset hypothesis is vacuous here and cannot make a competing prediction. Replicating the known irrep/Fourier signature on this run validates the measurement tools against a known answer; only the same-character-table pairs below can adjudicate between the hypotheses.
* **Calibration complete: the signature replicates, causally.** Three frequency blocks hold 94% of the embedding's energy (14–23× the random baseline); ablating any one costs 9–17 nats of test loss while the other 53 blocks sit at a 0.05-nat noise floor; the model restricted to just those three blocks retains 97.4% accuracy — all predicted in the [research log](docs/research-log.md) before the analysis ran. **Full write-up: [reports/01-c113-calibration.md](reports/01-c113-calibration.md).**

![Isotypic energy across training: diffuse during memorisation, concentrating into three frequency blocks exactly at the grokking transition](docs/figures/c113-energy-trajectory.png)

### The designed experiment: same-character-table pairs

Pairs of **non-isomorphic groups with identical character tables** make character-level evidence — the entire class of evidence the debate currently rests on — mathematically incapable of distinguishing the groups. Trained on such a pair, the hypotheses are forced apart:

| Pair | Order | Dataset size | What differs |
|---|---|---|---|
| Dih(104) vs Dic(104) | 104 | 10,816 | 53 vs 1 involutions; Dih has 52 reflection subgroups while every subgroup of Dic contains its unique involution; 2-dim irreps **real** vs **quaternionic** |
| Heisenberg group over F₅ vs C₂₅ ⋊ C₅ | 125 | 15,625 | element orders {1, 5} vs {1, 5, 25}; entirely different subgroup lattices |

That each pair shares a character table is not cited from the literature — it is computed and verified directly with the library's character-table machinery, and pinned by tests alongside the experiments.

Predictions: the **coset hypothesis** says learned-circuit statistics should track the (different) subgroup lattices; the **matrix-level irrep hypothesis** says structure should track the (different) irrep realisations — e.g. embedding rank ≈ 2 vs ≈ 4 inside the 2-dim isotypic block, since a quaternionic irrep has no 2×2 real realisation; a **character-level-only** account predicts no difference at all. The dihedral/dicyclic pair is primary: neither group is a direct product, so the task does not factor, and the real-vs-quaternionic contrast gives the irrep hypothesis its sharpest quantitative prediction; both are additionally contrasted with C₁₃ ⋊ C₈ (same order, different character table). The order-125 pair is the secondary comparison (each group is a central extension, so it likewise does not factor).

All of these are buildable with the existing machinery: the order-125 pair via the Todd–Coxeter presentation solver, Dih(104)/Dic(104) via the dihedral and dicyclic constructors, and C₁₃ ⋊ C₈ via `semidirect_product`.

### Results on the primary pair (Dih(104) vs Dic(104))

Across 38 seeds (weight decay 1.0; the matrix-level and coset contrasts taken on the 27 seeds where *both* groups grokked):

![Grokking epoch by group across 38 seeds: the dihedral group groks early and reliably, the dicyclic group later and often not at all](docs/figures/pair-grok-epochs.png)

* **The pair separates on *learnability*, not on a matrix-level signature.** Dih groks at **35/38** seeds (mean ~20k epochs, with **none** stuck in memorisation — the 3 misses are near-threshold at 0.94–0.99); Dic groks at **29/38**, much later (mean ~40k), with **6** seeds stuck in pure memorisation (test acc < 0.5). The quaternionic group is reliably harder to learn — a sub-character difference, since the character tables are identical.
* **Cosets add nothing over irreps.** Coset-membership decodability, scored against the model's *own* kept irreps (the control both prior papers omit), has **mean excess ≈ −0.05** across every proper normal subgroup and seed — zero or negative. The naive probe hits 100%, but so does the irrep control, which is what exposes it as vacuous.
* **No matrix-level real-vs-quaternionic signature.** The matrix-vs-trace R² gap — the instrument built to detect it — does **not** separate the groups (Welch p = 0.25 at n = 27).

On this pair, **irreps are sufficient and cosets add nothing**, and the cleanest discriminator is optimisation difficulty, not converged-weight structure. **Full write-up: [reports/02-irreps-vs-cosets.md](reports/02-irreps-vs-cosets.md).**

### Methods

Projections of embedding/attention weights onto isotypic (irreducible-representation) components via the projector library in `representations/`, coset/subgroup-alignment metrics scored against an irrep-restricted control, the matrix-vs-trace functional-form fit, per-component ablations, and SVD of weights across training checkpoints. A fully-connected baseline (in progress) controls for architecture (the coset evidence is FC-based, the irrep evidence transformer-based).

### Why this matters

The Chughtai/Stander disagreement is a clean instance of the central epistemic problem in mechanistic interpretability: **two incompatible mechanistic explanations fitting the same evidence**. If we cannot reliably adjudicate competing explanations in a 1-layer transformer on a fully-characterised algebraic task — where we can compute the ground-truth structure exactly — claims about circuits in frontier models rest on weak foundations. This project treats the toy setting as a testbed for evidence standards: what measurements, controls, and ablations does it actually take to confirm one mechanism over another?

---

## Status

| | |
|---|---|
| Done | Group algebra + representation-theory library (character tables, isotypic projectors, Todd–Coxeter); reproducible training pipeline (manifests, dual logging, event-dense checkpointing); C₁₁₃ grokking validation + irrep analysis ([report 01](reports/01-c113-calibration.md)); order-<20 negative result; checkpoint loading + activation-cache analysis API; functional-form fit + coset metrics with an irrep-restricted control; the Dih(104)/Dic(104) pair experiment across 38 seeds ([report 02](reports/02-irreps-vs-cosets.md)) |
| Active | Order-125 dim-5 go/no-go (Heisenberg/F₅ — does a richer irrep structure grok at all?); fully-connected baseline on the pair (architecture confound) |
| Planned | Order-125 second pair (C₂₅ ⋊ C₅); cross-run evaluation harness |

---

## Quick Start

```bash
uv sync

# Reproduce the C113 grokking run (~10 min on CPU, deterministic) ...
uv run python scripts/run.py data.group=C113 data.train_frac=0.3 optim.epochs=30000

# ... then run the full irrep analysis on it (energy spectra, ablations, trajectory)
uv run python scripts/analyze_run.py runs/<date>/<run_id>

# Reproduce the same-character-table pair (one matched setting shown; sweep seeds for more)
uv run python scripts/run.py data.group=D52   data.train_frac=0.4 optim.weight_decay=1.0 optim.epochs=80000
uv run python scripts/run.py data.group=Dic26 data.train_frac=0.4 optim.weight_decay=1.0 optim.epochs=80000

# Cross-seed comparison: learnability + matrix-level + coset tiers
uv run python scripts/compare_pairs.py --coset runs/<date> [runs/<date> ...]
```

Every run writes `manifest.json` (git hash, config hash, environment), `resolved_config.yaml`, `metrics.jsonl`, and weight checkpoints to `runs/<date>/<run_id>/`; the analysis adds `analysis/metrics.json` and figures. Any config field can be overridden with dotted CLI args, e.g. `data.train_frac=0.4 optim.weight_decay=1.0 experiment.seed=1`. (For a 5-second smoke test of the install, `data.group=C8 optim.epochs=200` works — but note the order-<20 finding above: groups that small memorise rather than generalise.)

---

## Engineering Hygiene & Reproducibility

The `training/` module enforces:
* **Configuration Validation:** Explicit schemas (via Pydantic) to validate experiment parameters before model initialization.
* **Deterministic Tracking:** Cryptographic run manifests (recording Git hashes, workspace state, and package versions) to guarantee replication.
* **Structured Telemetry:** Concurrent logging of scalar metrics to local structured JSONL files and Weights & Biases (W&B).
* **Exception Safety:** Automatic traceback capturing and run status marking if a training script encounters an unhandled exception.
