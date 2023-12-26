# services/data_service/data_handler_service.py
import pandas as pd
import numpy as np
import torch


class DataHandler:
    """
    A class for handling data preprocessing and scaling tasks.

    Attributes:
        scaler: An instance of a scaler class from sklearn or a custom scaler.
    """

    def __init__(self, scaler, original_df,  data_processor=None):
        """
        Initializes the DataHandler with the specified scaler.

        Args:
            scaler: An instance of a scaler class (e.g., StandardScaler from sklearn).
        """
        self.scaler = scaler
        self.prediction_df = None
        self.X_test = None
        self.X_test_tensor = None
        self.reference_df = None

    def preprocess_data(self, original_df):
        """
        Performs preprocessing on the provided DataFrame.

        Args:
            df (pandas.DataFrame): The data to be preprocessed.

        Returns:
            pandas.DataFrame: The preprocessed data.
        """
        self.reference_df = original_df
        inputs_scaled = self.create_data_loaders_from_dataframe(df=self.prediction_df)
        self.X_test_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

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
