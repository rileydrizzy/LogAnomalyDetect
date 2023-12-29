"""
Preprocessing Utility Module

Functions:
- clean_text(text_row: str) -> str: Preprocesses and cleans a text row.

- label_encoder(target_df: str) -> int: Performs label encoding for target labels.

- preprocess_and_encode(file_path: str, save_path: str): Preprocesses and encodes data,\
    saving the result.

- get_vectorization_layer(dataset: Dataset, max_length: Optional[int] = None) -> Tuple[TextVectorization, int]: Creates a text vectorization layer.

- convert_label_to_float(feature: str, label: int) -> Tuple[str, tf.Tensor]: Converts labels to float.

- get_dataset(file_path: str, batch_size: int = 2, shuffle_size: int = 100, shuffle: bool = False) -> Dataset: Creates a TensorFlow dataset with batching and prefetching.

- main(cfg: DictConfig): Main function for preprocessing data according to the provided configuration.

Example Usage:
    ```python
    # Preprocess and encode data
    preprocess_and_encode(file_path="raw_data.parquet", save_path="processed_data.parquet")
    
    # Create a TensorFlow dataset
    dataset = get_dataset(file_path="processed_data.parquet", batch_size=32, shuffle=True)
    ```

Note:
    The module assumes the existence of a configuration file named 'config.yaml' with the required parameters.
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


def clean_text(text_row: str):
    """Preprocesses and cleans a text row.

    Returns:
    - str: A cleaned and preprocessed text.
    """

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


def label_encoder(target_df: str):
    """Performs label encoding for target labels.

    Returns:
    - int: Encoded label (0 for 'normal', 1 for 'abnormal').
    """

    if target_df == "normal":
        label = 0
    else:
        label = 1
    return label


def preprocess_and_encode(file_path: str, save_path: str):
    """Preprocesses and encodes data, saving the result.

    Parameters:
    - file_path (str): Path of the input parquet file.
    - save_path (str): Path to save the processed data.
    """
    dataframe = pl.read_parquet(file_path)
    dataframe = dataframe.with_columns(
        pl.col("Target").apply(label_encoder, return_dtype=pl.Int16)
    )
    dataframe = dataframe.with_columns(pl.col("Log").apply(clean_text))
    dataframe.write_parquet(file=save_path, compression="gzip")


def get_vectorization_layer(dataset: tf.data.Dataset, max_length: int = None):
    """Creates a text vectorization layer.

    Parameters:
    - dataset (tf.data.Dataset): TensorFlow dataset containing text data.
    - max_length (Optional[int]): Maximum sequence length, by default None.

    Returns:
    - Tuple[tf.keras.layers.TextVectorization, int]: Vectorization layer and vocabulary size.
    """
    log_ds = dataset.map(lambda text, label: text)
    vectorization_layer = tf.keras.layers.TextVectorization(
        split="whitespace", output_mode="int", output_sequence_length=20
    )
    vectorization_layer.adapt(log_ds)
    vocab_size = vectorization_layer.vocabulary_size()

    return vectorization_layer, vocab_size


def convert_label_to_float(feature: str, label: int):
    """Converts labels to float.

    Parameters:
    - feature (str): Text feature.
    - label (int): Integer label.

    Returns:
    - Tuple[str, tf.Tensor]: Text feature and float label.
    """
    return feature, tf.cast(label, tf.float32)


def get_dataset(
    file_path: str, batch_size: int = 2, shuffle_size: int = 100, shuffle: bool = False
):
    """Creates a TensorFlow dataset with batching and prefetching.

    Parameters:
    - file_path (str): Path of the parquet file.
    - batch_size (int): Batch size.
    - shuffle_size (int): Size of the buffer for shuffle.
    - shuffle (bool): Perform shuffle on the dataset, by default False.

    Returns:
    - tf.data.Dataset: A TensorFlow Dataset with features and label.
    """
    dataframe = pl.read_parquet(file_path)
    features_df = dataframe["Log"].to_numpy()
    target_df = dataframe["Target"].to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((features_df, target_df))
    # dataset = dataset.map(convert_label_to_float)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main Preprocessing Function

    Parameters
    ----------
    cfg : DictConfig
        Configuration settings provided by Hydra.

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
