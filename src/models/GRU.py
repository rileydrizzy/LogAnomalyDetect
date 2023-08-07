"""doc
"""
from functools import partial

import tensorflow as tf

FILTER = 10
KERNEL = 5
STRIDE = 1

LR = 0.001
OPTIM = tf.keras.optimizers.Adam(learning_rate=LR)
LOSS = "binary_crossentropy"


def build_model(embed_dim, vocab_size, pad, sequence_length, tokenizer_layer):
    """1DCNN doc

    Parameters
    ----------
    file_path : str

    Returns
    -------
    model : object
        model
    """
    input_layer = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(None, sequence_length)), tokenizer_layer]
    )
    embeding_layer = tf.keras.layers.Embedding(
        input_dim=(vocab_size + 1), output_dim=embed_dim, mask_zero=True
    )
    DefaultConv1D = partial(
        tf.keras.layers.Conv1D, kernel_size=3, strides=1, padding=pad, activation="relu"
    )
    DefualtMaxpool1D = partial(tf.keras.layers.MaxPool1D, pool_size=2)
    model = tf.keras.Sequential(
        [
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
    main_model = tf.keras.Sequential([input_layer, model])
    main_model.compile(loss=LOSS, optimizer=OPTIM, metrics=["f1_score"])
    return main_model
