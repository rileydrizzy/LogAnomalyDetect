"""
Preprocessing Pipeline and Utility Module

This module defines functions related to a preprocessing pipeline for data
and includes utility functions.
The preprocessing steps include cleaning the 


Functions:
- clean_text(text_row: str) -> str: Preprocesses and cleans a text row.

- label_encoder(label: str) -> int: Performs label encoding for target labels.

- preprocess_and_encode(file_path: str, save_path: str): Preprocesses and encodes data,\
    saving the result.

- get_vectorization_layer(dataset: Dataset, max_length: int = 20): \
    Creates a text vectorization layer.

- convert_label_to_float(feature: str, label: int): Converts labels to float.

- get_dataset(file_path: str, batch_size: int = 2, \
    shuffle_size: int = 100, shuffle: bool = False): \
    Creates a TensorFlow dataset with batching and prefetching.

- main(cfg: DictConfig): Main function for preprocessing data according to the provided \
    configuration

Example:
    ```bash
    python src/dataset_loader.py
    ```
Note:
    The module assumes the existence of a configuration file named 'config.yaml'\
        with the required parameters.
"""

import re
import string

import hydra
import nltk
import polars as pl
import tensorflow as tf
from nltk.corpus import stopwords
from omegaconf import DictConfig

from utils.logging import logger

nltk.download("stopwords")


def clean_text(text_row):
    """Preprocesses and cleans a text row.

    Parameters
    ----------
    text_row : str
        The input text to be cleaned.

    Returns
    -------
    str
        A cleaned and preprocessed text.
    """

    if not text_row:
        return ""

    text_row = text_row.lower()
    text_row = re.sub("<[^>]*>", "", text_row)
    text_row = re.sub(r"[^a-zA-Z\s]", "", text_row)
    stop_words = set(stopwords.words("english"))
    text_row = [
        word
        for word in text_row.split()
        if word not in stop_words and word not in string.punctuation
    ]
    cleaned_text = " ".join(word for word in text_row)
    return cleaned_text


def label_encoder(label):
    """Performs label encoding for target labels.

    Parameters
    ----------
    label : str
        The target label to be encoded.

    Returns
    -------
    int
        Encoded label (0 for 'normal', 1 for 'abnormal').

    Raises
    ------
    ValueError
        If the label is not recognized, indicating an unexpected or invalid value.
    """
    if label == "normal":
        label = 0
    elif label == "abnormal":
        label = 1
    else:
        raise ValueError(f"Unrecognized label: {label}")

    return label


def preprocess_and_encode(file_path, save_path):
    """Preprocesses and encodes data, saving the result.

    Parameters
    ----------
    file_path : str
        Path of the input parquet file.
    save_path : str
        Path to save the processed data.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the specified file at `file_path` is not found.
    IOError
        If there is an error writing processed data to the specified `save_path`.
    """

    try:
        dataframe = pl.read_parquet(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"File not found: {file_path}") from error

    # Apply label encoding and text cleaning
    dataframe = dataframe.with_columns(
        pl.col("Target").apply(label_encoder, return_dtype=pl.Int16)
    )
    dataframe = dataframe.with_columns(pl.col("Log").apply(clean_text))

    try:
        dataframe.write_parquet(file=save_path, compression="gzip")
    except Exception as error:
        raise IOError(
            f"Error writing processed data to {save_path}: {error}"
        ) from error


def get_vectorization_layer(dataset, max_length=20):
    """Creates a text vectorization layer.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset containing text data.
    max_length : int, optional
         Maximum sequence length, by default 20.

    Returns
    -------
    Tuple[tf.keras.layers.TextVectorization, int]
        Vectorization layer and vocabulary size.
    """
    log_ds = dataset.map(lambda text, label: text)
    vectorization_layer = tf.keras.layers.TextVectorization(
        split="whitespace", output_mode="int", output_sequence_length=max_length
    )
    vectorization_layer.adapt(log_ds)
    vocab_size = vectorization_layer.vocabulary_size()

    return (vectorization_layer, vocab_size)


def convert_label_to_float(feature, label):
    """Converts labels to float.

    Parameters
    ----------
    feature : str
        Text feature.
    label : int
        Integer label.

    Returns
    -------
    Tuple[str, tf.Tensor]
        Text feature and float label.
    """
    return feature, tf.cast(label, tf.float32)


def get_dataset(file_path, batch_size=2, shuffle_size=100, shuffle=False):
    """
    Creates a TensorFlow dataset with batching and prefetching.

    Parameters
    ----------
    file_path : str
        Path of the parquet file.
    batch_size : int, optional
        Batch size, by default 2.
    shuffle_size : int, optional
        Size of the buffer for shuffle, by default 100.
    shuffle : bool, optional
        Perform shuffle on the dataset, by default False.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow Dataset with features and label.

    Raises
    ------
    FileNotFoundError
        If the specified file at `file_path` is not found.
    ValueError
        If one or more required columns specified in `required_columns`\
            are not present in the DataFrame.
    """

    try:
        dataframe = pl.read_parquet(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"File not found: {file_path}") from error

    # error handling for missing columns
    required_columns = {"Log", "Target"}
    if not set(dataframe.columns) >= required_columns:
        raise ValueError(
            f"Required columns {required_columns} not present in the DataFrame."
        )

    dataframe = dataframe.with_columns(dataframe["Target"].cast(pl.Float32))
    features_df = dataframe["Log"].to_numpy()
    target_df = dataframe["Target"].to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((features_df, target_df))

    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    # dataset = dataset.map(convert_label_to_float)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main Preprocessing Function

    Parameters
    ----------
    cfg : DictConfig
        Configuration settings provided by Hydra.
    
    Raises
    ------
    Exception
        If an unexpected error occurs during the preprocessing stage,\
            the exception is logged with details.

    Notes
    -----
    The function performs preprocessing and saving of train, validation, and test data.
    The processed data is saved according to the specified file paths in the configuration.
    """
    try:
        logger.info("Commencing preprocessing and saving of Train data")
        preprocess_and_encode(
            file_path=cfg.files.raw.raw_train_data,
            save_path=cfg.files.processed.train_dataset,
        )
        logger.success("Train data has been preprocessed and saved")

        logger.info("Commencing preprocessing and saving of Valid data")
        preprocess_and_encode(
            file_path=cfg.files.raw.raw_valid_data,
            save_path=cfg.files.processed.valid_dataset,
        )
        logger.success("Valid data has been preprocessed and saved")

        logger.info("Commencing preprocessing and saving of Test data")
        preprocess_and_encode(
            file_path=cfg.files.raw.raw_test_data,
            save_path=cfg.files.processed.test_dataset,
        )
        logger.success("Test data has been preprocessed and saved")
    except Exception as error:
        logger.exception(f"Preprocessing stage failed due to {error}")


if __name__ == "__main__":
    main()
    logger.success("Data has been processed, cleaned, and saved")
