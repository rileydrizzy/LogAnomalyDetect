"""train and validation on model and log result 

"""
#MLflow, tensorflowboard and Hypertuning setup

import hydra
from omegaconf import DictConfig
from dataset import testing_func
from utils import get_dataset

@hydra.main(version_base=None, config_path="conf", config_name="config")
#train_dataset = get_dataset()
#valid_dataset = get_dataset()
