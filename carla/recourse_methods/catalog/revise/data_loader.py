import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class dfDataset(Dataset):
    """
    Reads dataframe where last column is the label and the other columns are the features.
    The features are normalized by sklearn StandardScaler
    """

    def __init__(self, df: pd.DataFrame):
        self.columns = df.columns
        # all columns except last
        x = df.iloc[:, :-1].values
        # only last column
        y = df.iloc[:, -1:].values
        # scale to: mean zero, SD one
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(x)
        # x = self.scaler.transform(x)

        # Pandas
        self.x = x
        self.y = y
        # PyTorch
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.Y_train = torch.tensor(y)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_df(self, x=None, y=None, columns=None):
        """
        Returns read csv file as DataFrame with normalized features
        :param x:
        :param y:
        :param columns:
        :return:
        """
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if columns is None:
            columns = self.columns
        x = pd.DataFrame(data=x, columns=columns[:-1])
        y = pd.DataFrame(data=y, columns=columns[-1:])
        return pd.concat([x, y], axis=1)

    def convert_sample(self, df):
        """
        Returns the DataFrame with all columns, except the last one, normalized
        :param df:
        :return:
        """
        # all columns except last
        x = df.iloc[:, :-1].values
        # only last column
        y = df.iloc[:, -1:].values
        x = self.scaler.transform(x)
        return self.get_df(x, y, df.columns)
