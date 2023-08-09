"""doc
"""
from functools import partial

import tensorflow as tf


def build_model(embed_dim, vocab_size, pad):
    """1DCNN doc

    Parameters
    ----------
    file_path : str

    Returns
    -------
    model : object
        model
    """
    input_layer = tf.keras.layers.Input(shape=(20,))
    embeding_layer = tf.keras.layers.Embedding(
        input_dim=(vocab_size + 1),
        output_dim=embed_dim,
        input_length=20,
        mask_zero=True,
    )
    DefaultConv1D = partial(
        tf.keras.layers.Conv1D, kernel_size=3, strides=1, padding=pad, activation="relu"
    )
    DefualtMaxpool1D = partial(tf.keras.layers.MaxPool1D, pool_size=2)
    model = tf.keras.Sequential(
        [
            input_layer,
            embeding_layer,
            DefaultConv1D(30),
            DefualtMaxpool1D(),
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=20, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )
    return model
