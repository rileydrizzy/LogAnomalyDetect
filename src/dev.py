from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig):
    print(cfg.files.raw_data)
    print(cfg.files.json_file)
    print(cfg.files.valid_dataset)
my_app()
