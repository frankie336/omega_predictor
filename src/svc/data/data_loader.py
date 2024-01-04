import torch
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderService:
    """
    A service class for converting pandas DataFrames into PyTorch DataLoader objects.
    """

    def __init__(self, scaler_parameters, batch_size=32):
        """
        Initializes the DataLoaderService with scaler parameters and batch size.

        Args:
            scaler_parameters: Pre-computed parameters for scaling (e.g., a numpy array with mean and std).
            batch_size (int): The size of the batches to provide when loading data.
        """
        self.scaler_parameters = scaler_parameters
        self.batch_size = batch_size

    def apply_scaling(self, data):
        """
        Manually applies scaling to the data using the provided scaler parameters.

        Args:
            data (numpy.ndarray): The data to be scaled.

        Returns:
            numpy.ndarray: The scaled data.
        """
        # Assuming scaler_parameters is a tuple of (mean, std) for each feature
        mean, std = self.scaler_parameters
        return (data - mean) / std

    def create_data_loader_from_dataframe(self, df):
        """
        Creates DataLoader objects for the dataset contained in a pandas DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the dataset.

        Returns:
            DataLoader: A DataLoader containing the scaled and tensor-converted features.
        """
        # Convert DataFrame to numpy array
        data_array = df.values

        # Manually apply scaling
        scaled_data = self.apply_scaling(data_array)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(features_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return data_loader
