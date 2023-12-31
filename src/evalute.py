"""doc
"""

import hydra
import mlflow
import tensorflow as tf
from omegaconf import DictConfig

from models.model_loader import ModelLoader
from dataset_loader import get_dataset, get_vectorization_layer
from utils.common_utils import set_seed
from utils.logging import logger


def retrive_trained_model(model_name):
    client = mlflow.MlflowClient()
    version = client.get_latest_versions(name=model_name)[0].version
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.keras.load_model(model_uri)
    return model


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Parameters
    ----------
    cfg : DictConfig
        _description_
    """
    try:
        logger.info("Commencing evaluation process with test dataset")
        set_seed()

        logger.info("Retrieving the test dataset")
        train_data = get_dataset(
            file_path=cfg.files.processed.test_dataset, batch_size=cfg.params.batch_size
        )

        logger.info(f"Retrieving the model: {cfg.model_name}")

    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    main()
