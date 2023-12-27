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

Note:
    Ensure the required Python packages are installed before running the script:
    ```bash
    pip install mlflow tensorflow hydra-core
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
callbacks_list = [checkpoints_cb, early_stopping_cb]


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
        loss = tf.keras.losses.BinaryCrossentropy()
        optim = tf.keras.optimizers.Adam(learning_rate=cfg.params.learning_rate)

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
                model.compile(loss=loss, optimizer=optim)
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

    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    main()
