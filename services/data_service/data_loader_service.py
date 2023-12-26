import torch
from torch.utils.data import TensorDataset, DataLoader


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
        self.scaler = scaler
        self.batch_size = batch_size

    def create_data_loaders_from_dataframe(self, df, target_column):
        """
        Creates DataLoader objects for the dataset contained in a pandas DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the dataset.
            target_column (str): The name of the column to use as the target for predictions.

        Returns:
            DataLoader: A DataLoader containing the scaled and tensor-converted features and targets.
        """
        # Separate the features and the target
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        target_tensor = torch.tensor(y, dtype=torch.float32)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(features_tensor, target_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return data_loader
