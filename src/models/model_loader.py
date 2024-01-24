"""Model Loader Module

This module defines a `ModelLoader` class responsible for loading different TensorFlow models 
based on their names. The available models and their associated build functions are specified in 
the `models` dictionary.

Classes:
    - `ModelLoader`: Class for loading TensorFlow models.

Methods:
    - `get_model(model_name: str) -> object`: Builds and retrieves a model instance based on \
        the provided model name.

Note:
    Ensure that the desired models and their corresponding build functions are correctly defined \
    in the `models` dictionary within the `ModelLoader` class.

"""

from models import baseline_cnn


class ModelLoader:
    """
    Class for Loading TensorFlow Models
    """

    def __init__(self):
        self.models = {"11DCNN": baseline_cnn.build_model}

    def get_model(self, model_name: str) -> object:
        """Build and Retrieve a TensorFlow Model Instance.

        Parameters:
        - model_name (str): Name of the model.

        Returns:
        - object: The built model instance.

        Raises:
        - ValueError: If the specified model is not in the model list.
        """
        if model_name in self.models:
            return self.models[model_name]
        raise ValueError(
            f"Model '{model_name}' is not in the model list. Available models:\
                {list(self.models.keys())}"
        )
