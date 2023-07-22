""" utils
"""

import polars as pl
import tensorflow as tf

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
    features_df = dataframe['Log']
    target_df = dataframe['Target']
    dataset = tf.data.Dataset.from_tensor_slices((features_df, target_df))
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def class_weight_calc():
    """
    doc
    """
    pass 


def testing_func():
    """return train and valid data
    """
    print('getting data')
