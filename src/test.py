from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    print(cfg.paths.data)
    print(cfg.files.dev_data)
if __name__ == "__main__":
    my_app()