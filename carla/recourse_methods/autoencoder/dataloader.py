import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """
    Reads dataframe where last column is the label and the other columns are the features.
    """

    def __init__(self, data: np.ndarray, with_target=True):

        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if with_target:
            # all columns except the last
            self.X_train = torch.tensor(data[:, :-1], dtype=torch.float32).to(device)
            # only last column
            self.Y_train = torch.tensor(data[:, -1:]).to(device)
        else:
            # all columns
            self.X_train = torch.tensor(data, dtype=torch.float32).to(device)

        self.with_target = with_target

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        if self.with_target:
            return self.X_train[idx], self.Y_train[idx]
        else:
            return self.X_train[idx]
