""" Loads, clean, encode and export the data into train, valid and test 

Data(parquet file) is read and splitted into train, valid and test datasets using a 80:10:10 ratio. 
The data was splitted using stratfied steps to account for the class imbalanced 
The datasets(train, vaild and test set) was then cleaned and target label was encoded into integers 
and the datasets was then save as a parquet for efficent storage

functions:
    *load_split - load and split the dataset into train, valid and test set
    *clean_text_po - preforms text processing steps to clean the log text
    *label_encoder - perform label encoding on target
    *save_to_parquet - save and export dataset as parquet 
    *main - the main function of the script

"""
# import hydra and replace dir

import re
import string

import nltk
import polars as pl
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords")


def load_spit(file_path, target):
    """create dataframe and split the data into train, valid and test set,
    in a 80:10:10 ratio.
    Stati

    Parameters
    ----------
    file_path : str, Path object
        The file path of the file
    target : str
        The column name of the target

    Returns
    -------
    new_data : tuple
        A tuple containing the train_data, valid_data, test_data
    """

    datarame = pl.read_parquet(file_path)

    train_data, data = train_test_split(
        datarame, random_state=42, test_size=0.2, stratify=datarame[target]
    )
    valid_data, test_data = train_test_split(
        data, random_state=42, test_size=0.5, stratify=data[target]
    )
    new_data = (train_data, valid_data, test_data)
    return new_data


def clean_text_preprocess(text_row):
    """performs preprocessing  steps on each text row
    removing numbers, stopwords, punctuation and any

    Parameters
    ----------
    text_row : str
        text

    Returns
    -------
    clean_text : s
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
    clean_text = " ".join(word for word in text_row)
    return clean_text


def label_encoder(target_df):
    """performs label encoding for binary class

    Parameters
    ----------
    target_df : str
        Dataframe of the target

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


def save_to_parquet(dataframe, file_path):
    """save the dataframe as a parquet file

    Parameters
    ----------
    dataframe : str
        A dataframe
    file_path : str. Path object
        The path where to export the parquet

    Returns
    -------
        None
    """
    dataframe.write_parquet(file=file_path, compression="gzip")


def _dataset_func(file_path, batch_size, shuffle_size, shuffle=True):
    """create a Tensorflow dataset, with shuffle, batching and prefetching activated
    to speed up computation

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
    dataset = dataset.batch(batch_size).prefetch(1)  # interleave??
    return dataset
