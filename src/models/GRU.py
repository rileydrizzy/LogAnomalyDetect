"""doc
"""
from functools import partial

import tensorflow as tf


def build_model(vocab_size, embed_dim, Sequnce_length):
    """1DCNN doc

    Parameters
    ----------
    file_path : str

    Returns
    -------
    model : object
        model
    """

    input_ = tf.keras.layers.Input(shape=(Sequnce_length,))
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                output_dim=embed_dim, mask_zero= True)
    conv1D = tf.keras.layers.Conv1D(filters=10, kernel_size= 2)
    pool = tf.keras.layers.MaxPool1D()
    flatten = tf.keras.layers.GlobalAveragePooling1D()
    drop1 = tf.keras.layers.Dropout(0.5)
    dense_layer = tf.keras.layers.Dense(units =100, activation='relu')
    drop2 = tf.keras.layers.Dropout(0.5)
    output_layer = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.Sequential([input,embedding_layer,conv1D,pool,flatten,drop1,dense_layer,drop2,output_layer])
    return model