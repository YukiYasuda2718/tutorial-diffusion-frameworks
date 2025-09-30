import dataclasses

from src.configs.base_config import BaseConfig
from src.frameworks.ddpm import DDPMConfig


@dataclasses.dataclass()
class ExperimentLorenz96Config(BaseConfig):
    batch_size: int
    ddpm: DDPMConfig
