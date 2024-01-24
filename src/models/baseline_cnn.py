"""Baseline 1D Convolutional Network

This module defines a baseline 1D convolutional neural network using TensorFlow/Keras.

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
    """Builds a 1D Convolutional Neural Network model.

    Parameters
    ----------
    filter_num : int, optional
        Number of filters in the convolutional layer, by default 10.
    kernel_size_ : int, optional
        Size of the convolutional kernel, by default 2.
    activation_ : str, optional
        Activation function, by default "relu".
    embedding_vocab : int, optional
        Size of the vocabulary for the embedding layer, by default None.
    embedding_dim : int, optional
        Dimensionality of the embedding space, by default 32.
    vectorization_layer : tf.keras.layers.Layer, optional
        Layer for vectorization, by default None.
    pre_trained_embed : bool, optional
        Whether to use pre-trained embeddings, by default False.

    Returns
    -------
    tf.keras.Sequential
        A 1D Convolutional Neural Network model.

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
