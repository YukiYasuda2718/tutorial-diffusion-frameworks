import dataclasses

from src.configs.base_config import BaseConfig
from src.frameworks.ddpm import DDPMConfig


@dataclasses.dataclass()
class ExperimentLorenz96Config(BaseConfig):
    batch_size: int
    loss_name: str
    learning_rate: float
    #
    n_features: int
    list_channel: list[int]
    #
    total_epochs: int
    save_interval: int
    use_auto_mix_precision: bool
    ddpm: DDPMConfig
