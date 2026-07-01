# Command-line entry for experiment runs, with OmegaConf dotted overrides.
from typing import Any

from omegaconf import OmegaConf

from same_character_table_interp.training.config import GrokkingConfig
from same_character_table_interp.training.trainer import GroupGrokkingTrainer


def build_config(overrides: list[str]) -> GrokkingConfig:
    """Turn ``key=value`` CLI overrides into a validated GrokkingConfig."""
    cli = OmegaConf.from_dotlist(overrides)
    raw = OmegaConf.to_container(cli, resolve=True) or {}
    # from_dotlist always yields a mapping at the top level; narrow the union
    # OmegaConf declares (dict | list | str | None) so the kwargs below check.
    if not isinstance(raw, dict):
        raise TypeError(f"Overrides must form a mapping, got {type(raw).__name__}")
    container: dict[str, Any] = {str(key): value for key, value in raw.items()}

    # experiment.name/seed have no pydantic defaults, so supply them here.
    experiment = container.setdefault("experiment", {})
    experiment.setdefault("seed", 0)
    group = container.get("data", {}).get("group", "C8")
    experiment.setdefault("name", f"grok-{group}")

    return GrokkingConfig(**container)


def main(overrides: list[str]) -> None:
    config = build_config(overrides)
    trainer = GroupGrokkingTrainer.from_config(config)
    print(f"running {trainer.run_id} | group={config.data.group} | device={trainer.device}")
    final = trainer.fit()
    print("final:", {k: round(v, 4) for k, v in final.items()})
    print("run dir:", trainer.run_dir)
