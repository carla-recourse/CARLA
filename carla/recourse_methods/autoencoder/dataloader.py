import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """
    Reads dataframe all the columns are the features.
    """

    def __init__(self, data: np.ndarray):
        x = data

        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.tensor(x, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx]
