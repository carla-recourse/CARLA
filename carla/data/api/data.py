from abc import ABC, abstractmethod


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user. This is the general data object
    that is used in CARLA.
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
    def df(self):
        """
        The full Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Data transformation, for example normalization of continuous features and encoding of categorical features.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.Dataframe
        """
        pass

    @abstractmethod
    def inverse_transform(self, df):
        """
        Inverts transform operation.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.Dataframe
        """
        pass
