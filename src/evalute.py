"""Evaluation Script for Trained Models

This script performs evaluation on a trained TensorFlow model using a test dataset.\
    It retrieves the model either from local storage or MLflow based on the configuration.

Functions:
    - retrieve_trained_model(model_name: str, from_mlflow: bool = False) -> tf.keras.Model:\
        Retrieves a trained model either from local storage or MLflow.
    - main(cfg: DictConfig): Main function for the evaluation script.

Parameters:
    - `cfg` (DictConfig): Configuration parameters loaded using Hydra.

Notes:
    - The script evaluates the Precision-Recall Curve and Confusion Matrix on a trained model\
        using a test dataset.
    - It uses MLflow for model tracking if specified in the configuration.
Example:
    ```bash
    python src/evalute.py
    ```
"""

import hydra
import mlflow
import tensorflow as tf
from omegaconf import DictConfig

from dataset_loader import get_dataset
from utils.common_utils import (plot_confusion_matrix,
                                plot_precision_recall_curve, set_seed)
from utils.logging import logger


def retrieve_trained_model(model_name, from_mlflow=False):
    """
    Retrieves a trained model either from local storage or MLflow.

    Parameters:
    - `model_name` (str): Name of the model.
    - `from_mlflow` (bool): Flag indicating whether to retrieve the model from MLflow.\
        Default is False.

    Returns:
    - `tf.keras.Model`: Trained TensorFlow model.
    """
    if from_mlflow:
        client = mlflow.MlflowClient()
        version = client.get_latest_versions(name=model_name)[0].version
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.keras.load_model(model_uri)
        return model
    else:
        model_checkpoint = f"artifacts/{model_name}/model_checkpoints"
        model = tf.keras.models.load_model(model_checkpoint)
        return model


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function for the evaluation script.

    Parameters:
        - `cfg` (DictConfig): Configuration parameters loaded using Hydra.
    """
    try:
        logger.info("Commencing evaluation process with the test dataset")
        set_seed()

        f1_score_metrics = tf.keras.metrics.F1Score(
            average=None,
            threshold=None,
            name="f1_score",
        )
        logger.info("Retrieving the test dataset")
        test_data = get_dataset(
            file_path=cfg.files.processed.test_dataset, batch_size=cfg.params.batch_size
        )

        logger.info(f"Retrieving the model: {cfg.model_name}")
        model = retrieve_trained_model(cfg.model_name, from_mlflow=False)

        logger.info(
            f"Starting evaluation of Precision-Recall Curve on trained {cfg.model_name}"
        )
        plot_precision_recall_curve(
            model, model_name=cfg.model_name, eval_dataset=test_data, plot_label="Test"
        )
        logger.info("Precision-Recall Curve evaluation completed.")

        logger.info(
            f"Starting evaluation of the Confusion Matrix on trained {cfg.model_name}"
        )
        plot_confusion_matrix(
            model,
            model_name=cfg.model_name,
            eval_dataset=test_data,
            threshold=cfg.params.cm_threshold,
            plot_label="Test",
        )
        logger.info("Confusion Matrix evaluation completed.")

    except ValueError:
        logger.error(
            "Plotting of the Confusion Matrix and Precision-Recall Curve failed due to empty data"
        )
    except Exception as error:
        logger.exception(f"Evaluation failed due to -> {error}.")


if __name__ == "__main__":
    main()
