import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .config_schema import BaseConfig


def compute_config_hash(config: BaseConfig) -> str:
    config_dict = config.model_dump()
    sorted_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(sorted_json.encode()).hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return len(result) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_run_id(experiment_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{experiment_name}"


def create_run_dir(run_id: str) -> Path:
    date_str = run_id.split("_")[0]
    run_dir = Path("runs") / date_str / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_manifest(config: BaseConfig, run_dir: Path) -> dict:
    run_id = run_dir.name
    config_hash = compute_config_hash(config)
    # Record the configured device (what the run used) and whether torch
    # deterministic algorithms were active -- both matter for reproducibility.
    device = getattr(config.experiment, "device", None) or detect_device()

    manifest = {
        "run_id": run_id,
        "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.experiment.name,
        "seed": config.experiment.seed,
        "device": device,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "available_device": detect_device(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "status": "running",
        "end_time": None,
        "final_metrics": None,
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def update_manifest(
    run_dir: Path,
    status: str,
    final_metrics: dict | None = None,
    error: str | None = None,
) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = status
    manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    if final_metrics is not None:
        manifest["final_metrics"] = final_metrics
    if error is not None:
        manifest["error"] = error
    manifest_path.write_text(json.dumps(manifest, indent=2))


def save_resolved_config(config: BaseConfig, run_dir: Path) -> None:
    config_path = run_dir / "resolved_config.yaml"
    config_dict = config.model_dump()
    config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(config_dict)))
