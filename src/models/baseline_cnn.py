"""Baseline 1D Convulotional Network

"""
import tensorflow as tf


def build_model(
    filter_num=10,
    kernel_size_=2,
    activation_="relu",
    embedding_vocab=None,
    embedding_dim=32,
    vectorization_layer=None,
):
    """_summary_

    Parameters
    ----------
    filter_num : int, optional
        _description_, by default 10
    kernel_size_ : int, optional
        _description_, by default 2
    activation_ : str, optional
        _description_, by default "relu"
    embedding_vocab : int, optional
        _description_, by default None
    embedding_dim : int, optional
        _description_, by default 32
    vectorization_layer : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    input_layer = tf.keras.Input(shape=(1,), dtype=tf.string, name="input_layer")
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=embedding_vocab, output_dim=embedding_dim
    )
    conv1d_layer = tf.keras.layers.Conv1D(
        filters=filter_num, kernel_size=kernel_size_, activation=activation_
    )
    pool_layer = tf.keras.layers.MaxPool1D(strides=2)
    flatten_layer = tf.keras.layers.Flatten()
    dropout_1 = tf.keras.layers.Dropout(rate=0.1)
    dense_layer = tf.keras.layers.Dense(10, activation=activation_)
    dropout_2 = tf.keras.layers.Dropout(rate=0.5)
    classifier = tf.keras.layers.Dense(1, activation="sigmoid")
    model = tf.keras.Sequential(
        [
            input_layer,
            vectorization_layer,
            embedding_layer,
            conv1d_layer,
            pool_layer,
            flatten_layer,
            dropout_1,
            dense_layer,
            dropout_2,
            classifier,
        ]
    )
    return model
