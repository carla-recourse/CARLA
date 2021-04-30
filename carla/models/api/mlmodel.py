from abc import ABC, abstractmethod


class MLModel(ABC):
    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : List of String
            Correct order of input features for ml model
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """
        Describes the type of backend which is used for the ml model.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        backend : String
            Used framework
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Returns the raw ml model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1].

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.Array
            Ml model prediction for interval [0, 1] with shape N x 1
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : float
            Ml model prediction with shape N x 2
        """
        pass

    @property
    @abstractmethod
    def scaler(self):
        """
        Yields a fitted normalizer.
        To keep a consistent and correct normalization for the ML model

        We recommend to use sklear normalization preprocessing

        Returns
        -------
        Fitted scaler for normalization
        """
        pass

    @abstractmethod
    def set_scaler(self, data):
        """
        Sets and fits the correct normalizer.

        Parameters
        ----------
        data : carla.data.Data()
            Contains the data to fit a scaler

        Returns
        -------

        """
        pass

    @property
    @abstractmethod
    def encoder(self):
        """
        Yields a fitted encoding function.
        To keep consistent and correct encoding for the ML model.

        We recommend to use the sklearn OneHotEncoding

        Returns
        -------
        Fitted encoder
        """
        pass

    @abstractmethod
    def set_encoder(self, data):
        """
        Sets and fits the correct encoder.

        Parameters
        ----------
        data : carla.data.Data()
            Contains the data to fit an encoder

        Returns
        -------

        """
        pass
