import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """
    Reads dataframe where last column is the label and the other columns are the features.
    The features are normalized by sklearn StandardScaler
    """

    def __init__(self, data: np.ndarray):
        # all columns except last
        x = data[:, :-1]
        # only last column
        y = data[:, -1:]

        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.tensor(x, dtype=torch.float32).to(device)
        self.Y_train = torch.tensor(y).to(device)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]
