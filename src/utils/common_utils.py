""" 
Utility Functions Module

Functions:
- set_seed(): 
"""

import os
from pathlib import Path
from time import strftime

import mlflow
import tensorflow as tf


def set_seed(seed=42):
    """_summary_

    Parameters
    ----------
    seed : int, optional
        _description_, by default 42
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
    """_summary_

    Parameters
    ----------
    name : str
        _description_

    Returns
    -------
    int
        _description_
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    experiment = mlflow.get_experiment_by_name(model_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(model_name)
        return experiment_id
    return experiment.experiment_id


def tensorboard_dir(model_name):
    """_summary_

    Parameters
    ----------
    model_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    model_directory = "Tensorbord_logs/" + model_name
    runs = strftime("run_%Y_%m_%d_%H_%M_%S")
    log_dir = Path(model_directory, runs)
    return log_dir


def get_device_strategy():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    if tf.test.gpu_device_name():
        strategy = tf.distribute.MirroredStrategy()
        return strategy, "GPU"
    strategy = tf.distribute.OneDeviceStrategy(device="/device:CPU:0")
    return strategy, "CPU"
