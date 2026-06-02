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
    └── finite-groups/              # How networks learn finite group operations
```

---

## Projects

### Active

#### How Neural Networks Learn Finite Group Operations
* **Question (broad, narrowing):** When a network learns to compose elements of a finite group, *what algorithm and internal representation does it use* — and how does that depend on the group's structure and the model architecture? This engages an open debate in the literature: do networks compose via the group's **irreducible representations** ([Chughtai et al., 2023](https://arxiv.org/abs/2302.03025)) or via **coset / subgroup structure** ([Stander et al., 2024](https://arxiv.org/abs/2312.06581))? Grokking is one phenomenon studied along the way, not the sole focus.
* **Approach:** Build finite groups (a complete catalog of order < 20, plus larger groups via presentations / Todd–Coxeter), train small models (a 1-layer attention+MLP transformer, with a fully-connected baseline planned) on the multiplication task, and reverse-engineer the learned computation.
* **Interpretability methods:** projections of embedding/attention weights onto isotypic (irreducible-representation) components, planned coset/subgroup-alignment metrics, singular value decomposition of weights over training, and bilinear-asymmetry analysis. The specific question will be locked in as the evidence points somewhere sharp.

---

## Engineering Hygiene & Reproducibility

To ensure experimental rigour, the experiments utilize a shared `core/` library that enforces:
* **Configuration Validation:** Explicit schemas (via Pydantic) to validate experiment parameters before model initialization.
* **Deterministic Tracking:** Cryptographic run manifests (recording Git hashes, workspace state, and package versions) to guarantee replication.
* **Structured Telemetry:** Concurrent logging of scalar metrics to local structured JSONL files and Weights & Biases (W&B).
* **Exception Safety:** Automatic traceback capturing and run status marking if a training script encounters an unhandled exception.
