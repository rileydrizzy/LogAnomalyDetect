"""
Utility Functions Module

Functions:
- set_seed(seed=42): Set random seed for reproducibility.

- set_mlflow_tracking(model_name): Set up MLflow experiment tracking.

- tensorboard_dir(model_name): Generate directory for TensorBoard logs.

- get_device_strategy(): Get TensorFlow distribution strategy based on available devices.
"""

import os
from pathlib import Path
from time import strftime

import mlflow
import tensorflow as tf


def set_seed(seed=42):
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value, by default 42.
    """
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def set_mlflow_tracking(model_name):
    """Set up MLflow experiment tracking.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    int
        Experiment ID for MLflow tracking.

    Notes
    -----
    The function assumes a DagsHub-specific MLflow tracking URI.
    """
    mlflow.set_tracking_uri("https://dagshub.com/rileydrizzy/log_anomaly.mlflow")
    experiment = mlflow.get_experiment_by_name(model_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(model_name)
        return experiment_id
    return experiment.experiment_id


def tensorboard_dir(model_name):
    """Generate directory for TensorBoard logs.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    pathlib.Path
        Path to the directory for TensorBoard logs.
    """
    model_directory = "Tensorbord_logs/" + model_name
    runs = strftime("run_%Y_%m_%d_%H_%M_%S")
    log_dir = Path(model_directory, runs)
    return log_dir


def get_device_strategy():
    """Get TensorFlow distribution strategy based on available devices.

    Returns
    -------
    tuple
        - tf.distribute.Strategy: TensorFlow distribution strategy.
        - str: Description of the available device ('GPU' or 'CPU').
    """
    if tf.test.gpu_device_name():
        strategy = tf.distribute.MirroredStrategy()
        return (strategy, "GPU")
    strategy = tf.distribute.OneDeviceStrategy(device="/device:CPU:0")
    return (strategy, "CPU")
