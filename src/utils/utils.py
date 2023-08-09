""" utils
"""
import os
from pathlib import Path
from time import strftime

import mlflow
import polars as pl
import tensorflow as tf


def get_dataset(file_path, batch_size=2, shuffle_size=100, shuffle=False):
    """create a Tensorflow dataset, with shuffle, batching and prefetching activated
    to speed up computation during training

    Parameters
    ----------
    file_path : str
        path of the parquet file
    batch_size : int
        Batch size
    shuffle_size : int
        Size of the buffer for shuffle
    shuffle : bool, Default = True
        perform shuffle on the dataset, if false it doesn't

    Returns
    -------
    dataset : Dataset
        A tensorflow Dataset with features and label
    """
    dataframe = pl.read_parquet(file_path)
    features_df = dataframe["Log"]
    target_df = dataframe["Target"]
    dataset = tf.data.Dataset.from_tensor_slices((features_df, target_df))
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


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


def get_tokenizer(dataset):
    """_summary_

    Parameters
    ----------
    dataset : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    log_ds = dataset.map(lambda text, label: text)
    tokenizer_layer = tf.keras.layers.TextVectorization(
        split="whitespace", output_mode="int", output_sequence_length=20
    )
    tokenizer_layer.adapt(log_ds)
    vocab_size = tokenizer_layer.vocabulary_size()

    return tokenizer_layer, vocab_size


def tracking(name):
    """_summary_

    Parameters
    ----------
    name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    mlflow.set_tracking_uri("https://dagshub.com/rileydrizzy/dogbreeds_dect.mlflow")
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        return experiment_id
    return experiment.experiment_id


def tensorboard(model_name):
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
