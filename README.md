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
├── scripts/run.py          # experiment launcher
├── tests/                  # 150 tests incl. mathematical property tests
└── runs/                   # local run artifacts (gitignored)
```

---

## The Research Question

When a network learns to compose elements of a finite group, *what algorithm and internal representation does it use* — and how does that depend on the group's structure? This engages an open debate in the literature: do networks compose via the group's **irreducible representations** ([Chughtai et al., 2023](https://arxiv.org/abs/2302.03025)) or via **coset / subgroup structure** ([Stander et al., 2024](https://arxiv.org/abs/2312.06581))?

The published evidence on both sides is **character-level** (correlations between model outputs/weights and irreducible characters) — and both hypotheses fit that evidence. Neither paper varied the *group* to a regime where the hypotheses make divergent predictions. That is the gap this project targets.

### Findings so far

* **Groups of order < 20 do not generalise.** Across S₃, Q₈, A₄, C₈ × weight-decay sweeps (150k epochs), every model memorised the training set quickly and stayed at chance test accuracy. The dataset is the bottleneck (|G|² ≤ 361 examples), not optimisation — so small groups are learning a lookup table, not the group operation, and any "algorithm" read off them would be an artefact. The investigation therefore runs on groups of order ≈ 100–350.
* **Pipeline validated at scale.** C₁₁₃ (modular addition, the canonical grokking task) groks cleanly: train_frac 0.3, 30k epochs → test accuracy **99.98%**, with the weight-norm progress measure and dense checkpoints captured through the transition.

### The designed experiment: same-character-table pairs

Pairs of **non-isomorphic groups with identical character tables** make character-level evidence — the entire class of evidence the debate currently rests on — mathematically incapable of distinguishing the groups. Trained on such a pair, the hypotheses are forced apart:

| Pair | Order | Dataset size | What differs |
|---|---|---|---|
| Dih(104) vs Dic(104) | 104 | 10,816 | 53 vs 1 involutions; different subgroup lattices; 2-dim irreps **real** vs **quaternionic** |
| Heisenberg group over F₅ vs C₂₅ ⋊ C₅ | 125 | 15,625 | element orders {1, 5} vs {1, 5, 25}; entirely different subgroup lattices |

That each pair shares a character table is not cited from the literature — it is computed and verified directly with the library's character-table machinery, and pinned by tests alongside the experiments.

Predictions: the **coset hypothesis** says learned-circuit statistics should track the (different) subgroup lattices; the **matrix-level irrep hypothesis** says structure should track the (different) irrep realisations — e.g. embedding rank ≈ 2 vs ≈ 4 inside the 2-dim isotypic block, since a quaternionic irrep has no 2×2 real realisation; a **character-level-only** account predicts no difference at all. The dihedral/dicyclic pair is primary: neither group is a direct product, so the task does not factor, and the real-vs-quaternionic contrast gives the irrep hypothesis its sharpest quantitative prediction; both are additionally contrasted with C₁₃ ⋊ C₈ (same order, different character table). The order-125 pair is the secondary comparison (each group is a central extension, so it likewise does not factor).

All of these are buildable with the existing machinery: the order-125 pair via the Todd–Coxeter presentation solver, Dih(104)/Dic(104) via the dihedral and dicyclic constructors, and C₁₃ ⋊ C₈ via `semidirect_product`.

### Methods

Projections of embedding/attention weights onto isotypic (irreducible-representation) components via the projector library in `representations/`, coset/subgroup-alignment metrics (in progress), per-component ablations, and SVD of weights across training checkpoints. A fully-connected baseline is planned to control for architecture (the coset evidence is FC-based, the irrep evidence transformer-based).

### Why this matters

The Chughtai/Stander disagreement is a clean instance of the central epistemic problem in mechanistic interpretability: **two incompatible mechanistic explanations fitting the same evidence**. If we cannot reliably adjudicate competing explanations in a 1-layer transformer on a fully-characterised algebraic task — where we can compute the ground-truth structure exactly — claims about circuits in frontier models rest on weak foundations. This project treats the toy setting as a testbed for evidence standards: what measurements, controls, and ablations does it actually take to confirm one mechanism over another?

---

## Status

| | |
|---|---|
| Done | Group algebra + representation-theory library (character tables, isotypic projectors, Todd–Coxeter); reproducible training pipeline (manifests, dual logging, event-dense checkpointing); C₁₁₃ grokking validation; order-<20 negative result; checkpoint loading + activation-cache analysis API (`analysis/`) |
| Active | Irrep-level analysis of the grokked C₁₁₃ run: isotypic-energy spectra, per-block ablations, grokking-transition trajectories |
| Planned | Coset/subgroup-alignment metrics; the same-character-table pair experiment; FC baseline; cross-run evaluation harness |

---

## Quick Start

```bash
uv sync
uv run python scripts/run.py data.group=C8 optim.epochs=2000
```

Every run writes `manifest.json` (git hash, config hash, environment), `resolved_config.yaml`, `metrics.jsonl`, and weight checkpoints to `runs/<date>/<run_id>/`. Any config field can be overridden with dotted CLI args, e.g. `data.group=A4 data.train_frac=0.4 optim.weight_decay=1.0 experiment.seed=1`.

---

## Engineering Hygiene & Reproducibility

The `training/` module enforces:
* **Configuration Validation:** Explicit schemas (via Pydantic) to validate experiment parameters before model initialization.
* **Deterministic Tracking:** Cryptographic run manifests (recording Git hashes, workspace state, and package versions) to guarantee replication.
* **Structured Telemetry:** Concurrent logging of scalar metrics to local structured JSONL files and Weights & Biases (W&B).
* **Exception Safety:** Automatic traceback capturing and run status marking if a training script encounters an unhandled exception.
