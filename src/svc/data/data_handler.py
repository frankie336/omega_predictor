# services/data/data_handler.py
import pandas as pd
import torch
from svc.data.data_loader import DataLoaderService


class DataHandler:
    """
    A class for handling data preprocessing and scaling tasks.

    Attributes:
        scaler: An instance of a scaler class from sklearn or a custom scaler.
    """

    def __init__(self, scaler, original_df):
        """
        Initializes the DataHandler with the specified scaler.

        Args:
            scaler: An instance of a scaler class (e.g., StandardScaler from sklearn).
        """
        self.scaler = scaler
        DataLoaderService(scaler=self.scaler)
        self.create_data_loaders_from_dataframe = DataLoaderService.create_data_loaders_from_dataframe(df=self.prediction_df)
        self.prediction_df = None
        self.X_test = None
        self.X_test_tensor = None
        self.reference_df = None

    def preprocess_data(self, original_df):
        """
        Processes the feature engineered data to create tensors for prediction. It scales the data using
        the 'create_data_loaders_from_dataframe' method and then converts it into a PyTorch tensor.

        Args:
            original_df (pandas.DataFrame): The original DataFrame containing the feature-engineered data.

        Returns:
            torch.Tensor: A tensor representing the scaled and processed input data ready for prediction.
        """
        self.reference_df = original_df
        inputs_scaled = self.create_data_loaders_from_dataframe
        self.X_test_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
        return self.X_test_tensor

    def scale_data(self, data):
        """
        Scales the data using the provided scaler.

        Args:
            data (pandas.DataFrame or numpy.ndarray): The data to be scaled.

        Returns:
            numpy.ndarray: The scaled data.
        """
        inputs = data.values if isinstance(data, pd.DataFrame) else data
        inputs_scaled = self.scaler.fit_transform(inputs)

        return inputs_scaled
