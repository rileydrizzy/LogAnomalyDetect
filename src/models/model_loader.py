"""doc
"""

from models.baseline_cnn import build_model


class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {"1DCNN": build_model, "test_model": None}

    def get_model(self, model_name):
        """build and retrieve the model instance

        Parameters
        ----------
        model_name : str
            model name

        Returns
        -------
        object
            return built model instance
        """

        if model_name in self.models:
            return self.models[model_name]
        raise ValueError("Model is not in the model list")
