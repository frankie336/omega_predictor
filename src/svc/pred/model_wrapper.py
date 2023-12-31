# svc/pred/model_wrapper.py
import torch
from svc.model.simple_classifier import SimpleBinaryClassifier


class ModelWrapper:
    """
    A wrapper class for the machine learning model to handle prediction and processing.
    """
    def __init__(self, loaded_model):
        """
        Initializes the ModelWrapper with a pre-loaded model.

        Args:
            loaded_model (torch.nn.Module): A pre-loaded and initialized neural network model.
        """
        self.model = loaded_model
        self.device = loaded_model.device  # Assuming the model has a 'device' attribute

    def load_model(self, model_path, input_size):
        """
        Initializes the ModelWrapper with the specified model and input size.

        Args:
            model_path (str): The path to the saved model file.
            input_size (int): The number of input features the model expects.
        """
        model = SimpleBinaryClassifier(input_size=input_size)  # Pass any required arguments

        # Load the state dictionary into the model
        state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(state_dict)

        return model

    def predict(self, data):
        # Make predictions using the preprocessed tensor data
        if data is not None:
            # Ensure the model is in evaluation mode
            self.model.eval()

            # Perform prediction
            with torch.no_grad():
                prediction_probabilities = torch.sigmoid(self.model(data)).numpy()

            # Determine the dimension of output probabilities and reshape accordingly
            if prediction_probabilities.ndim == 1 or prediction_probabilities.shape[1] == 1:
                probability_positive_class = prediction_probabilities.flatten()
            elif prediction_probabilities.shape[1] == 2:
                # Two-dimensional with separate probabilities for each class
                probability_positive_class = prediction_probabilities[:, 1]
            else:
                # Handle unexpected shapes
                raise ValueError(f"Unexpected shape of prediction_probabilities: {prediction_probabilities.shape}")

            # Convert probabilities to binary predictions
            binary_predictions = (probability_positive_class > 0.5).astype(int)

            return binary_predictions, probability_positive_class
        else:
            print("No data provided for prediction.")
            return None  # Corrected indentation here


if __name__ == '__main__':
    model_Wrapper = ModelWrapper()