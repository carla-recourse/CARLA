from abc import ABC, abstractmethod


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user.
    """

    @property
    @abstractmethod
    def categoricals(self):
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
    def continous(self):
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
