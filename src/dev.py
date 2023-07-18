from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))

def test(cfg: DictConfig):
    print(cfg.)