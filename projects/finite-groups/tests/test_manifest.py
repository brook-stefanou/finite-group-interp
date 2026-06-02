import json

from core.config_schema import ExperimentConfig
from core.manifest import compute_config_hash, create_manifest
from finite_groups.experiments.config import GrokkingConfig, OptimConfig


def _config(epochs: int = 40000) -> GrokkingConfig:
    return GrokkingConfig(
        experiment=ExperimentConfig(name="t", seed=0),
        optim=OptimConfig(epochs=epochs),
    )


def test_manifest_embeds_resolved_config(tmp_path):
    # The manifest must be self-contained: it records the full resolved config,
    # not just a hash, so a run is reproducible from the manifest alone.
    create_manifest(_config(epochs=40000), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert "config" in manifest
    assert manifest["config"]["optim"]["epochs"] == 40000


def test_manifest_config_matches_recorded_hash(tmp_path):
    # The embedded config and the recorded hash must describe the same config.
    config = _config(epochs=12345)
    create_manifest(config, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["config_hash"] == compute_config_hash(config)
    assert manifest["config"]["optim"]["epochs"] == 12345
