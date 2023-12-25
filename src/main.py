"""doc
"""

import hydra
import mlflow
import tensorflow as tf
from omegaconf import DictConfig

from dataset_loader import get_dataset, get_vectorization_layer
from models.model_loader import ModelLoader
from utils.common_utils import (get_device_strategy, set_mlflow_tracking,
                                set_seed, tensorboard_dir)
from utils.logging import logger

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
    Parameters
    ----------
    cfg : DictConfig
        _description_
    """
    try:
        logger.info("Commencing training process")
        set_seed()
        experiment_id = set_mlflow_tracking(cfg.model_name)
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

        logger.info("Retrieving the training strategy.")
        strategy, device = get_device_strategy()
        logger.info(
            f"Training on {device}, with {strategy.num_replicas_in_sync} num_proc"
        )

        logger.info(f"Retrieving the model: {cfg.model_name}")
        load_model_func = ModelLoader().get_model(cfg.model_name)
        loss = tf.keras.losses.BinaryCrossentropy()
        optim = tf.keras.optimizers.Adam(learning_rate=cfg.params.learning_rate)

        mlflow.tensorflow.autolog(log_datasets=False)
        with mlflow.start_run(
            experiment_id=experiment_id,
        ):
            mlflow.set_tag("model_name", cfg.model_name)

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
                model, artifact_path=f"model_artifact/{cfg.model_name}"
            )
            logger.success("Training Job completed")
    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    main()
