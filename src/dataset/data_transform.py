"""doc
"""

import re
import string

import hydra
import nltk
import polars as pl
from nltk.corpus import stopwords
from omegaconf import DictConfig

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
    text_cleaned = " ".join(word for word in text_row)
    return text_cleaned


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
        pl.col("Target").apply(label_encoder, return_dtype=pl.Int32)
    )
    dataframe = dataframe.with_columns(pl.col("Log").apply(clean_text))
    dataframe.write_parquet(file=save_path, compression="gzip")


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """_summary_

    Parameters
    ----------
    cfg : DictConfig
        _description_
    """
    try:
        preprocess_and_encode(file_path=cfg.files.raw.raw_train_data, save_path=cfg.files.processed.train_dataset)
        preprocess_and_encode(file_path=cfg.files.raw.raw_valid_data, save_path=cfg.files.processed.valid_dataset)
        preprocess_and_encode(file_path=cfg.files.raw.raw_test_data, save_path=cfg.files.processed.test_dataset)
    except Exception:
        pass


if __name__ == "__main__":
    main()
