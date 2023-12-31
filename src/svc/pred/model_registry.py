import torch
from pathlib import Path
from src.svc.model.simple_classifier import SimpleBinaryClassifier


class ModelRegistryService:
    """
    Service class for managing and providing access to different machine learning models,
    supporting both local and Hugging Face-hosted models.
    """

    def __init__(self):
        self.models = {}  # A dictionary to store model instances
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_identifier, input_size, model_path=None):
        """
        Loads and returns the model instance based on the provided identifier.

        Args:
            model_identifier (str): A unique identifier for the model.
            input_size (int): The number of input features the model expects.
            model_path (str, optional): The path to the model's state dictionary. If not provided,
                                        the local path will be used.

        Returns:
            torch.nn.Module: The loaded and initialized model.
        """
        if model_identifier not in self.models:
            model = SimpleBinaryClassifier(input_size=input_size)
            state_dict_path = model_path or self.get_model_path(model_identifier)
            state_dict = torch.load(state_dict_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            self.models[model_identifier] = model
        return self.models[model_identifier]

    def get_model_path(self, model_identifier):
        """
        Retrieves the file path for the model's state dictionary based on its identifier.

        Args:
            model_identifier (str): A unique identifier for the model.

        Returns:
            str: The file path of the model's state dictionary.
        """
        root_dir_path = Path(__file__).resolve().parent
        model_dir = root_dir_path / "models"
        model_dir.mkdir(exist_ok=True)
        return model_dir / f"{model_identifier}.pt"

if __name__ == '__main__':
    model_registry_service = ModelRegistryService()
    model_identifier = 'IntraDayForexPredictor_v4.0.10'
    model_path = model_registry_service.get_model_path(model_identifier=model_identifier)
    print(model_path)
    model = model_registry_service.load_model(model_identifier=model_identifier, input_size=10)

