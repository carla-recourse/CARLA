from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from sklearn import preprocessing

from carla.models.pipelining import decode, descale, encode, scale


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user.
    """

    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def raw(self):
        """
        The raw Dataframe without encoding or normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with raw information
        """
        pass

    @property
    @abstractmethod
    def train_raw(self):
        """
        The raw training split Dataframe without encoding or normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with raw information
        """
        pass

    @property
    @abstractmethod
    def test_raw(self):
        """
        The raw testing split Dataframe without encoding or normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with raw information
        """
        pass

    @property
    @abstractmethod
    def scaler(self):
        """
        Contains a fitted sklearn scaler.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        pass

    @property
    @abstractmethod
    def encoder(self):
        """
        Contains a fitted sklearn encoder:

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        pass

    @abstractmethod
    def processed(self, with_target=True):
        """
        The processed Dataframe with encoding and normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with processed information
        """
        pass

    @abstractmethod
    def train_processed(self, with_target=True):
        """
        The processed training split Dataframe with encoding and normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with processed information
        """
        pass

    @abstractmethod
    def test_processed(self, with_target=True):
        """
        The processed testing split Dataframe with encoding and normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with processed information
        """
        pass

    def __fit_scaler(self, scaling_method):
        # TODO add StandardScaler
        # If needed another scaling method can be added here
        if scaling_method == "MinMax":
            fitted_scaler = preprocessing.MinMaxScaler().fit(self.raw[self.continuous])
        else:
            raise ValueError("Scaling Method not known")
        return fitted_scaler

    def __fit_encoder(self, encoding_method):
        if encoding_method == "OneHot":
            fitted_encoder = preprocessing.OneHotEncoder(
                handle_unknown="error", sparse=False
            ).fit(self.raw[self.categorical])
        elif encoding_method == "OneHot_drop_binary":
            fitted_encoder = preprocessing.OneHotEncoder(
                drop="if_binary", handle_unknown="error", sparse=False
            ).fit(self.raw[self.categorical])
        else:
            raise ValueError("Encoding Method not known")
        return fitted_encoder

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("encoder", lambda x: decode(self.encoder, self.categorical, x)),
            ("scaler", lambda x: descale(self.scaler, self.continuous, x)),
        ]
