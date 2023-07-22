"""
doc
"""

import tensorflow as tf
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")

def model(cfg: DictConfig):
    """doc
    """
    Input_layer = tf.keras.layers.Input(shape=(1,1))
    embeding_layer = tf.keras.layers.Embedding()


    model.complie(loss = ['BCE'],
                  class_weights = ,
                  optimizer= )
    return model
