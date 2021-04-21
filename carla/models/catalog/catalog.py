import numpy as np

from ..api import MLModel
from .load_model import load_model


class MLModelCatalog(MLModel):
    def __init__(
        self,
        data_name,
        model_type,
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
        data_name : str
            Name of the dataset which is used to predict [adult]
        model_type : str
            Architecture [ann]
        backend : str
            Specifies the used framework [tensorflow, pytorch]
        cache : boolean, optional
            If True, try to load from the local cache first, and save to the cache
            if a download is required.
        models_home : string, optional
            The directory in which to cache data; see :func:`get_models_home`.
        kws : keys and values, optional
            Additional keyword arguments are passed to passed through to the read model function
        """
        self._backend = backend

        if self._backend == "pytorch":
            ext = "pt"
        elif self._backend == "tensorflow":
            ext = "h5"
        else:
            raise Exception("Model type not in catalog")

        self._data_name = data_name
        self._model = load_model(model_type, data_name, ext, cache, models_home, **kws)

        if data_name == "adult":
            self._feature_input_order = [
                "age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "workclass_Private",
                "marital-status_Non-Married",
                "occupation_Other",
                "relationship_Non-Husband",
                "race_White",
                "sex_Male",
                "native-country_US",
            ]
        else:
            raise Exception("Model for dataset not in catalog")

    @property
    def feature_input_order(self):
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : List of String
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
        backend : String
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
        output : np.Array
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        if self._backend == "pytorch":
            return self._model.predict(x)
        elif self._backend == "tensorflow":
            return self._model.predict(x)[:, 1]
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

        if self._backend == "pytorch":
            class_1 = 1 - self._model.forward(x).detach().numpy().squeeze()
            class_2 = self._model.forward(x).detach().numpy().squeeze()

            # For single prob prediction it happens, that class_1 is casted into float after 1 - prediction
            # Additionally class_1 and class_2 have to be at least shape 1
            if not isinstance(class_1, np.ndarray):
                class_1 = np.array(class_1).reshape(1)
                class_2 = class_2.reshape(1)

            return np.array(list(zip(class_1, class_2)))

        elif self._backend == "tensorflow":
            return self._model.predict(x)
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )
