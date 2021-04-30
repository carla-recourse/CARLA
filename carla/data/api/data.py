from abc import ABC, abstractmethod


class Data(ABC):
    """
    Abstract architecture to allow arbitrary datasets, which are provided by the user.
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
        list : List of Strings
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
        list : List of Strings
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
        list : List of Strings
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
        String : String
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
        df : :class:`pandas.DataFrame`
            Tabular data with raw information
        """
        pass

    @property
    @abstractmethod
    def normalized(self):
        """
        The normalized Dataframe without encoding

        Type of normalization can be arbitrary

        Returns
        -------
        df : :class:`pandas.DataFrame`
            Tabular data with normalized information
        """
        pass

    @abstractmethod
    def set_normalized(self, mlmodel):
        """
        Normalizes the dataframe with respect to the normalization used for the ML model.
        Result is saved in self.normalized.

        Parameters
        ----------
        mlmodel : carla.model.MLModel
            ML model which contains information about normalization

        Returns
        -------

        """
        pass

    @property
    @abstractmethod
    def encoded(self):
        """
        The encoded Dataframe without normalization

        Type of encoding can be arbitrary

        Returns
        -------
        df : :class:`pandas.DataFrame`
            Tabular data with encoded information
        """
        pass

    @abstractmethod
    def set_encoded(self, mlmodel):
        """
        Encodes the dataframe with respect to the encoding used for the ML model
        Result is saved in self.encoded.

        Parameters
        ----------
        mlmodel : carla.model.MLModel
            ML model which contains information about encoding

        Returns
        -------

        """
        pass

    @property
    @abstractmethod
    def encoded_normalized(self):
        """
        The normalized and encoded Dataframe

        Type of normalization and encoding have to be the same as for normalized and encoded

        Returns
        -------
        df : :class:`pandas.DataFrame`
            Tabular data with normalized and encoded information
        """
        pass

    @abstractmethod
    def set_encoded_normalized(self, mlmodel):
        """
        Normalizes and encodes the dataframe with respect to the normalization/ encoding used for the ML model
        Result is saved in self.encoded_normalized, and if not already set, in self.encoded and self.normalized, too.

        Parameters
        ----------
        mlmodel : carla.model.MLModel
            ML model which contains information about normalization and encoding

        Returns
        -------

        """
        pass
