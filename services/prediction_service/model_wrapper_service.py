# services/prediction_service/model_wrapper_service.py
import torch
from models.deep_nn import SimpleBinaryClassifier

class ModelWrapper:
    def __init__(self, model_path, input_size):
        self.model = self.load_model(model_path, input_size)

    def load_model(self, model_path, input_size):
        # Initialize an instance of the model
        model = SimpleBinaryClassifier(input_size=input_size)  # Pass any required arguments

        # Load the state dictionary into the model
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
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
            return None, None
