""" doc

"""

import tensorflow as tf
import hydra
from omegaconf import DictConfig


def build_model():
    input_layer = tf.keras.Input