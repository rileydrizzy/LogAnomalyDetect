""" utils
"""

import string
import re
import nltk
import hydra
from omegaconf import DictConfig
import polars as pl
import tensorflow as tf
from nltk.corpus import stopwords
nltk.download('stopwords')

def clean_text_preprocess(text_row):
    """ performs preprocessing steps on each text row removing numbers, 
    stopwords, punctuation and any symbols

    Returns 
    -------
    clean_text : row
        A cleaned and preprocessed text 
    """

    text_row = text_row.lower()
    text_row = re.sub('<[^>]*>', '', text_row)
    text_row = re.sub(r'[^a-zA-Z\s]', '', text_row)
    stop_words = set(stopwords.words('english'))
    text_row = [word for word in text_row.split()
            if word not in stop_words and word not in string.punctuation]
    clean_text = ' '.join(word for word in text_row)
    return clean_text

def label_encoder(target_df):
    """performs label encoding for binary class

    Returns
    -------
    label : int
        return either 0 for normal or 1 for abnormal
    """

    if target_df == 'normal':
        label = 0
    else:
        label = 1
    return label

def get_dataset(file_path, batch_size, shuffle_size, shuffle = True):
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
    dataframe = dataframe.with_columns(pl.col('Target').apply(
            label_encoder, return_dtype=pl.Int32))
    dataframe = dataframe.with_columns(pl.col('Log').apply(
            clean_text_preprocess))
    features_df = dataframe['Log']
    target_df = dataframe['Target']
    dataset = tf.data.Dataset.from_tensor_slices((features_df, target_df))
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def testing_func():
    """return train and valid data
    """
    print('getting data')

tokenizer = tf.keras.layers.TextVectorization(split= 'whitespace', output_mode= 'int',
                                              output_sequence_length= sequence_lenght)

tf.keras.layers.Embedding(input_dim=, output_dim=, mask_zero= ,)