from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    use_wandb: bool = False
    wandb_project: str = "ai-safety-portfolio"
    wandb_entity: str | None = None


class BaseConfig(BaseModel):
    experiment: ExperimentConfig
