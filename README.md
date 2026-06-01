# AI Safety & Representation Geometry Portfolio

![CI](https://github.com/stefanoubrook/ai-safety-portfolio/actions/workflows/ci.yml/badge.svg)

A codebase for learning AI safety and machine learning interpretability techniques through self-directed replication and extension of algorithmic research.

The primary objective is to investigate representation geometry, feature mapping, and circuit formation in neural networks by building clean, reproducible experiments.

---

## Repository Structure

The repository is structured as a monorepo with a shared engineering module. Individual projects are entirely decoupled inside the `projects/` directory.

```
ai-safety-portfolio/
├── .gitignore
├── README.md
├── core/                           # Shared engineering module
│   ├── config_schema.py
│   ├── logging_jsonl.py
│   ├── manifest.py
│   ├── metrics.py
│   └── trainer.py
└── projects/                       # Independent research workspaces
    └── finite-groups/              # Active grokking experiment
```

---

## Projects

### Active

#### Representation Alignment in Finite Group Grokking
* **Goal:** Does the training dynamics of algorithmic networks implicitly solve the group's representation theory? Specifically, does the network's internal representation align with the group's irreducible representations, and does the phase transition of grokking correspond to the collapse of general weight matrices onto the spanned subspaces of these irreducible representations?
* **Scope:** Training 1-layer attention-only transformers over all finite groups of order less than 20, mapping the geometry of internal embedding matrices against their canonical irreducible characters.
* **Interpretability Methods:** Canonical projections of embedding and attention weights onto isotypic components, Singular Value Decomposition of weights over training time, and bilinear asymmetry analysis.

---

## Engineering Hygiene & Reproducibility

To ensure experimental rigour, the experiments utilize a shared `core/` library that enforces:
* **Configuration Validation:** Explicit schemas (via Pydantic) to validate experiment parameters before model initialization.
* **Deterministic Tracking:** Cryptographic run manifests (recording Git hashes, workspace state, and package versions) to guarantee replication.
* **Structured Telemetry:** Concurrent logging of scalar metrics to local structured JSONL files and Weights & Biases (W&B).
* **Exception Safety:** Automatic traceback capturing and run status marking if a training script encounters an unhandled exception.
