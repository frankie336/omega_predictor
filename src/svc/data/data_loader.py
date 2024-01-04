import torch
from torch.utils.data import DataLoader, TensorDataset

class DataLoaderService:
    """
    A service class for converting pandas DataFrames into PyTorch DataLoader objects.
    """

    def __init__(self, scaler, batch_size=32):
        """
        Initializes the DataLoaderService with a scaler and batch size.

        Args:
            scaler: An instance of a scaler class (e.g., StandardScaler from sklearn).
            batch_size (int): The size of the batches to provide when loading data.
        """
        if not hasattr(scaler, 'fit_transform'):
            raise ValueError("scaler must have a fit_transform method.")
        self.scaler = scaler
        self.batch_size = batch_size

    def create_data_loader_from_dataframe(self, df):
        """
        Creates DataLoader objects for the dataset contained in a pandas DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the dataset.

        Returns:
            DataLoader: A DataLoader containing the scaled and tensor-converted features.
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(df)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(features_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return data_loader

