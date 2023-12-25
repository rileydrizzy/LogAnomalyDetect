"""doc
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
    """performs preprocessing steps on each text row removing numbers,
    stopwords, punctuation and any symbols

    Returns
    -------
    clean_text : row
        A cleaned and preprocessed text
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


def label_encoder(target_df):
    """performs label encoding for target label

    Returns
    -------
    label : int
        return either 0 for normal or 1 for abnormal
    """

    if target_df == "normal":
        label = 0
    else:
        label = 1
    return label


def preprocess_and_encode(file_path, save_path):
    """_summary_

    Parameters
    ----------
    file_path : _type_
        _description_
    save_path : _type_
        _description_
    """
    dataframe = pl.read_parquet(file_path)
    dataframe = dataframe.with_columns(
        pl.col("Target").apply(label_encoder, return_dtype=pl.Int16)
    )
    dataframe = dataframe.with_columns(pl.col("Log").apply(clean_text))
    dataframe.write_parquet(file=save_path, compression="gzip")


def get_vectorization_layer(
    dataset,
    max_length=None,
):
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
    vectorization_layer = tf.keras.layers.TextVectorization(
        split="whitespace", output_mode="int", output_sequence_length=20
    )
    vectorization_layer.adapt(log_ds)
    vocab_size = vectorization_layer.vocabulary_size()

    return vectorization_layer, vocab_size


def convert_label_to_float(feature, label):
    """_summary_

    Parameters
    ----------
    feature : _type_
        _description_
    label : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return feature, tf.cast(label, tf.float32)


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
    """_summary_

    Parameters
    ----------
    cfg : DictConfig
        _description_
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
        logger.exception(f"Preporcessing stage failed due to {error}")


if __name__ == "__main__":
    main()
    logger.success("Data has been processed, cleaned and saved")
