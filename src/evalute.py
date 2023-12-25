"""doc
"""

import hydra
import mlflow
import tensorflow as tf
from omegaconf import DictConfig

from models.model_loader import ModelLoader
from src.dataset_loader import get_dataset, get_vectorization_layer
from utils.common_utils import set_seed
from utils.logging import logger


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
        experiment_id = set_mlflow_tracking(cfg.model_name)
        model_tensorb_dir = tensorboard_dir(cfg.model_name)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_tensorb_dir)
        callbacks_list.append(tensorboard_cb)

        logger.info("Retrieving the test dataset")
        train_data = get_dataset(
            file_path=cfg.files.processed.test_dataset, batch_size=cfg.params.batch_size
        )

        logger.info(f"Retrieving the model: {cfg.model_name}")
        load_model_func = ModelLoader().get_model(cfg.model_name)

        mlflow.tensorflow.autolog(log_datasets=False)
        with mlflow.start_run(
            experiment_id=experiment_id,
        ):
            mlflow.set_tag("model_name", cfg.model_name)
            tokenizer, vocab_size = get_vectorization_layer(dataset=train_data)
            model = load_model_func(
                vectorization_layer=tokenizer, embedding_vocab=vocab_size
            )
            model
        logger.success("Training Job completed")
    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    main()
