"""Training Script for TensorFlow Model

This script defines a TensorFlow training pipeline using MLflow for experiment tracking
 and DVC (Data Version Control) for managing model artifacts. 
 The training process includes distributed training, logging to TensorBoard, 
 and automatic logging of metrics with MLflow.

Modules:
    - `dataset_loader`: Module providing functions to load and preprocess datasets.
    - `model_loader`: Module for loading different TensorFlow models.
    - `common_utils`: Utilities for setting seeds, managing devices, and directory paths.
    - `logging`: Custom logging utilities.

Functions:
    - `main(cfg: DictConfig)`: Main function orchestrating the training process.

Usage:
    Run this script to initiate the training process.
    Configuration settings are managed through Hydra and defined in the 'config.yaml' file.

Example:
    ```bash
    python src/main.py
    ```
"""


import hydra
import mlflow
import tensorflow as tf
from omegaconf import DictConfig

# Importing functions and classes from other modules
from dataset_loader import get_dataset, get_vectorization_layer
from models.model_loader import ModelLoader
from utils.common_utils import (
    get_device_strategy,
    set_mlflow_tracking,
    set_seed,
    tensorboard_dir,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)
from utils.logging import logger

# Callbacks for model training
checkpoints_cb = tf.keras.callbacks.ModelCheckpoint(
    "model_checkpoints",
    save_best_only=True,
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, verbose=1
)
callbacks_list = [checkpoints_cb, early_stopping_cb, reduce_lr]

# Directory to save plots.
PLOTS_DIR = "docs/plots/training"


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function for training the TensorFlow model.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing model parameters.
    """
    try:
        logger.info("Commencing training process")

        # Setting a seed for reproducibility
        set_seed()

        # Set up MLflow tracking for the experiment
        experiment_id = set_mlflow_tracking(cfg.model_name)

        # Set up TensorBoard logging directory
        model_tensorb_dir = tensorboard_dir(cfg.model_name)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_tensorb_dir)
        callbacks_list.append(tensorboard_cb)

        logger.info("Retrieving the dataset for training and validation sets")
        train_data = get_dataset(
            file_path=cfg.files.processed.train_dataset,
            batch_size=cfg.params.batch_size,
            shuffle=True,
        )
        valid_data = get_dataset(
            file_path=cfg.files.processed.valid_dataset,
            batch_size=cfg.params.batch_size,
        )

        logger.info(" Retrieving the training strategy for distributed training")
        strategy, device = get_device_strategy()
        logger.info(
            f"Training on {device}, with {strategy.num_replicas_in_sync} num_proc"
        )

        logger.info(f"Retrieving the model: {cfg.model_name}")
        load_model_func = ModelLoader().get_model(cfg.model_name)
        loss_func = tf.keras.losses.BinaryCrossentropy()
        optim = tf.keras.optimizers.Adam(learning_rate=cfg.params.learning_rate)
        f1_score_metrics = tf.keras.metrics.F1Score(
            average=None,
            threshold=None,
            name="f1_score",
        )

        # Enable MLflow autologging
        mlflow.tensorflow.autolog(log_datasets=False)

        # Start MLflow run
        with mlflow.start_run(
            experiment_id=experiment_id,
        ):
            mlflow.set_tag("model_name", cfg.model_name)

            # Training the model within the distributed strategy scope
            with strategy.scope():
                tokenizer, vocab_size = get_vectorization_layer(dataset=train_data)
                model = load_model_func(
                    vectorization_layer=tokenizer, embedding_vocab=vocab_size
                )
                model.compile(
                    loss=loss_func,
                    optimizer=optim,
                    metrics=[f1_score_metrics()],
                )
                logger.info(
                    f" Training {cfg.model_name} for {cfg.params.total_epochs} epochs"
                )
                model.fit(
                    train_data,
                    validation_data=valid_data,
                    epochs=cfg.params.total_epochs,
                    callbacks=callbacks_list,
                    class_weight=None,
                )

            logger.info(f"Saving Trained {cfg.model_name} Model")
            mlflow.tensorflow.log_model(
                model,
                artifact_path=f"model_artifact/{cfg.model_name}",
                registered_model_name=f"{cfg.model_name}",
            )
            logger.success("Training Job completed")

            logger.info(
                f"Starting evaluation of Precision-Recall Curve on trained {cfg.model_name}"
            )
            plot_precision_recall_curve(
                model,
                eval_dataset=valid_data,
                save_path=f"{PLOTS_DIR}/{cfg.model_name}_PR.png",
            )
            logger.info("Precision-Recall Curve evaluation completed.")

            logger.info(
                f"Starting evaluation of Confusion Matrix on trained {cfg.model_name}"
            )
            plot_confusion_matrix(
                model,
                eval_dataset=valid_data,
                threshold=cfg.params.cm_threshold,
                save_path=f"{PLOTS_DIR}/{cfg.model_name}_CM.png",
            )
            logger.info("Confusion Matrix evaluation completed.")
    except ValueError:
        logger.critical(
            "Plotting of Confusion Matrix and Precision-Recall Curve failed due to empty data"
        )

    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    main()
