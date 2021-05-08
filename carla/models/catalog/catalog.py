import numpy as np
import pandas as pd
from sklearn import preprocessing

from ...data.api import data as data_api
from ..api import MLModel
from .load_model import load_model


class MLModelCatalog(MLModel):
    def __init__(
        self,
        data: data_api,
        model_type,
        feature_input_order,
        encoding,
        backend="tensorflow",
        cache=True,
        models_home=None,
        **kws
    ):
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch" and "tensorflow".
        Possible models are corrently "ann".

        Parameters
        ----------
        model_type : str
            Architecture [ann]
        feature_input_order : list
            List containing all features in correct order for ML prediction
        encoding : list
            List containing encoded features in the form of [feature-name]_[value]
        backend : str
            Specifies the used framework [tensorflow, pytorch]
        cache : boolean, optional
            If True, try to load from the local cache first, and save to the cache
            if a download is required.
        models_home : string, optional
            The directory in which to cache data; see :func:`get_models_home`.
        kws : keys and values, optional
            Additional keyword arguments are passed to passed through to the read model function
        data : data.api.Data Class
            Correct dataset for ML model
        """
        self._backend = backend
        self._data = data

        if self._backend == "pytorch":
            ext = "pt"
        elif self._backend == "tensorflow":
            ext = "h5"
        else:
            raise Exception("Model type not in catalog")

        self._model = load_model(model_type, data.name, ext, cache, models_home, **kws)

        self._feature_input_order = feature_input_order

        # Preparing pipeline components
        self._continuous = data.continous
        self._categoricals = data.categoricals
        self._scaler = preprocessing.MinMaxScaler().fit(data.raw[self._continuous])

        self._encodings = encoding

    def pipeline(self, df):
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to use to keep correct encodings, normalization and input order

        Parameters
        ----------
        df : pd.DataFrame
            Contains unnormalized and not encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input in correct order, normalized and encoded

        """
        output = df.copy()

        # Normalization
        output[self._continuous] = self._scaler.transform(output[self._continuous])

        # Encoding
        output[self._encodings] = 0
        for encoding in self._encodings:
            for cat in self._categoricals:
                if cat in encoding:
                    value = encoding.split(cat + "_")[-1]
                    output.loc[output[cat] == value, encoding] = 1
                    break

        # Get correct order
        output = output[self._feature_input_order]

        return output

    def need_pipeline(self, x):
        """
        Checks if ML model input needs pipelining.
        Only DataFrames can be used to pipeline input.

        Parameters
        ----------
        x : pd.DataFrame or np.Array

        Returns
        -------
        bool : Boolean
            True if no pipelining process is already taken
        """
        if not isinstance(x, pd.DataFrame):
            return False

        if x.select_dtypes(exclude=[np.number]).empty:
            return False

        return True

    @property
    def feature_input_order(self):
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : list of str
            Correct order of input features for ml model
        """
        return self._feature_input_order

    @property
    def backend(self):
        """
        Describes the type of backend which is used for the ml model.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        backend : str
            Used framework
        """
        return self._backend

    @property
    def raw_model(self):
        """
        Returns the raw ml model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        return self._model

    def predict(self, x):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1]

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        input = self.pipeline(x) if self.need_pipeline(x) else x

        if self._backend == "pytorch":
            return self._model.predict(input)
        elif self._backend == "tensorflow":
            return self._model.predict(input)[:, 1]
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

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

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        input = self.pipeline(x) if self.need_pipeline(x) else x

        if self._backend == "pytorch":
            class_1 = 1 - self._model.forward(input).detach().numpy().squeeze()
            class_2 = self._model.forward(input).detach().numpy().squeeze()

            # For single prob prediction it happens, that class_1 is casted into float after 1 - prediction
            # Additionally class_1 and class_2 have to be at least shape 1
            if not isinstance(class_1, np.ndarray):
                class_1 = np.array(class_1).reshape(1)
                class_2 = class_2.reshape(1)

            return np.array(list(zip(class_1, class_2)))

        elif self._backend == "tensorflow":
            return self._model.predict(input)
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    @property
    def data(self):
        return self._data
