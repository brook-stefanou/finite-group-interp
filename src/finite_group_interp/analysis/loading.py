"""Rebuild trained models from checkpoint files and run directories.

Checkpoints are self-contained: the trainer embeds the full resolved config in
every ``.pt`` payload, so a model can be reconstructed from the file alone --
the config names the group, the group fixes the vocab sizes, and
``build_model`` (the same recipe the trainer used) gives the architecture.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.groups.group import FiniteGroup
from finite_group_interp.model import OneLayerTransformer
from finite_group_interp.training.config import GrokkingConfig
from finite_group_interp.training.trainer import build_model


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: OneLayerTransformer  # weights loaded, eval() mode, CPU
    config: GrokkingConfig  # validated from the dict embedded in the .pt
    group: FiniteGroup
    epoch: int
    path: Path


def load_checkpoint(path: Path | str) -> LoadedCheckpoint:
    """Rebuild the model saved at ``path`` (a trainer ``.pt`` payload)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    required = {"config", "model_state_dict", "epoch"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError(
            f"{path} is not a trainer checkpoint: expected a payload dict with "
            f"keys {sorted(required)}"
        )
    # Extra fields in old configs (e.g. the deleted init_std) are ignored by
    # pydantic's default extra="ignore".
    config = GrokkingConfig.model_validate(payload["config"])
    group = resolve_group(config.data.group)
    model = build_model(config, group)
    # strict load, with one tolerated gap: pre-June-4 checkpoints predate the
    # causal_mask buffer, which __init__ rebuilds deterministically.
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    problems = [k for k in missing if k != "causal_mask"] + list(unexpected)
    if problems:
        raise RuntimeError(f"state_dict mismatch loading {path}: {problems}")
    model.eval()
    return LoadedCheckpoint(
        model=model, config=config, group=group, epoch=int(payload["epoch"]), path=path
    )


# Trailing integer in a checkpoint stem: step_N or grokked_step_N. The
# filename step is the epoch the trainer saved at, so ordering by it avoids
# torch-loading hundreds of files just to sort them.
_STEP_RE = re.compile(r"(\d+)$")


def _step_of(path: Path) -> int:
    match = _STEP_RE.search(path.stem)
    if match is None:
        raise ValueError(f"cannot parse a step number from checkpoint filename: {path.name}")
    return int(match.group(1))


def list_checkpoints(run_dir: Path | str) -> list[Path]:
    """All checkpoint files of a run, sorted by training step (ascending).

    Only trainer-written checkpoints (``step_N.pt`` / ``grokked_step_N.pt``)
    are listed; other ``.pt`` files are ignored. When ``step_N`` and
    ``grokked_step_N`` share a step, the grokked file sorts last, so it wins
    latest-checkpoint selection.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"no checkpoints directory in run dir: {run_dir}")
    matching = [p for p in ckpt_dir.glob("*.pt") if _STEP_RE.search(p.stem)]
    paths = sorted(matching, key=lambda p: (_step_of(p), p.stem.startswith("grokked")))
    if not paths:
        raise FileNotFoundError(f"no .pt checkpoints in {ckpt_dir}")
    return paths


@dataclass(frozen=True)
class LoadedRun:
    checkpoint: LoadedCheckpoint
    run_dir: Path
    metrics: list[dict[str, Any]]  # parsed metrics.jsonl, one dict per logged eval
    manifest: dict[str, Any]  # parsed manifest.json


def load_run(run_dir: Path | str, checkpoint: str | Path | None = None) -> LoadedRun:
    """Load a training run: model (latest checkpoint by default) + history.

    ``checkpoint`` selects a specific snapshot:

    * ``None`` – latest checkpoint (by training step; grokked beats plain at a
      tie).
    * ``str`` – checkpoint stem, e.g. ``"step_1024"``; resolved as
      ``<run_dir>/checkpoints/<stem>.pt``.
    * absolute ``Path`` – trusted as-is.
    * relative ``Path`` – resolved against the run's checkpoints directory
      (consistent with the stem form; CWD is not used).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    if checkpoint is None:
        ckpt_path = list_checkpoints(run_dir)[-1]
    elif isinstance(checkpoint, Path):
        # Absolute paths are trusted as-is; relative ones are taken as
        # run-relative (consistent with the stem form).
        ckpt_path = checkpoint if checkpoint.is_absolute() else run_dir / "checkpoints" / checkpoint
    else:
        ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    loaded = load_checkpoint(ckpt_path)
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return LoadedRun(checkpoint=loaded, run_dir=run_dir, metrics=metrics, manifest=manifest)
