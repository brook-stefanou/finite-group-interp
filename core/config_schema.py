from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    device: str = "cpu"  # cpu is deterministic and fast for small models; "mps"/"cuda" to override
    deterministic: bool = True  # enable torch deterministic algorithms for reproducible runs
    use_wandb: bool = False
    wandb_project: str = "ai-safety-portfolio"
    wandb_entity: str | None = None


class BaseConfig(BaseModel):
    experiment: ExperimentConfig
