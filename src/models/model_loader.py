"""doc
"""

from src.models import baseline_cnn

class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {"cnn": baseline_cnn.build_model, "test_model": None}

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
        else:
            raise ValueError("Model is not in the model list")
