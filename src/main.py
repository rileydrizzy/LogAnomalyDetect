import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(cfg)


main()
