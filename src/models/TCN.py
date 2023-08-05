"""
doc
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_name="models", config_path="config", version_base="1.2")
def test_func(cfg: DictConfig):
    print(cfg)
