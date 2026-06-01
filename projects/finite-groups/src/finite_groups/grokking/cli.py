# Command-line entry for grokking runs, with OmegaConf dotted overrides.
from omegaconf import OmegaConf

from finite_groups.grokking.config import GrokkingConfig
from finite_groups.grokking.trainer import GroupGrokkingTrainer


def build_config(overrides: list[str]) -> GrokkingConfig:
    """Turn ``key=value`` CLI overrides into a validated GrokkingConfig."""
    cli = OmegaConf.from_dotlist(overrides)
    container = OmegaConf.to_container(cli, resolve=True) or {}

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
