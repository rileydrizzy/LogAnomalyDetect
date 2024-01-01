"""doc
"""

import hydra
import mlflow
import tensorflow as tf
from omegaconf import DictConfig


from dataset_loader import get_dataset
from utils.common_utils import (
    set_seed,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)
from utils.logging import logger

# Directory to save plots.
plots_dir = "docs/plots/evaluation"


def retrive_trained_model(model_name, from_mlflow: False):
    """_summary_

    Parameters
    ----------
    model_name :str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if from_mlflow:
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
        model = retrive_trained_model(cfg.model_name)

        logger.info(
            f"Starting evaluation of Precision-Recall Curve on trained {cfg.model_name}"
        )
        plot_precision_recall_curve(
            model, eval_dataset=test_data, save_path=f"{plots_dir}/{cfg.model_name},"
        )
        logger.info("Precision-Recall Curve evaluation completed.")

        logger.info(
            f"Starting evaluation of Confusion Matrix on trained {cfg.model_name}"
        )
        plot_confusion_matrix(
            model,
            eval_dataset=test_data,
            threshold=cfg.params.cm_threshold,
            save_path=f"",
        )
        logger.info("Confusion Matrix evaluation completed.")

    except ValueError:
        logger.error(
            "Plotting of Confusion Matrix and Precision-Recall Curve failed due to empty data"
        )
    except Exception as error:
        logger.exception(f"Evaluation failed due to -> {error}.")


if __name__ == "__main__":
    main()
