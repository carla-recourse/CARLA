import numpy as np
import pandas as pd
import torch

from carla.data.load_catalog import load_catalog
from carla.models.pipelining import encode, order_data, scale

from ..api import MLModel
from .load_model import load_model


class MLModelCatalog(MLModel):
    def __init__(
        self,
        data,
        model_type,
        backend="tensorflow",
        cache=True,
        models_home=None,
        use_pipeline=False,
        **kws
    ):
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch" and "tensorflow".
        Possible models are corrently "ann".

        Parameters
        ----------
        data : data.api.Data Class
            Correct dataset for ML model
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
        use_pipeline : bool, optional
            If true, the model uses a pipeline before predict and predict_proba to preprocess the input data.
        """
        super().__init__(data)
        self._backend = backend

        # TODO: Add logic for feature_order depending on upcoming different catalog models
        self._catalog = load_catalog(data.catalog_file, data.name)
        if self._backend == "pytorch":
            ext = "pt"
            self._feature_input_order = self._catalog["feature_input_order_bin"]
        elif self._backend == "tensorflow":
            ext = "h5"
            self._feature_input_order = self._catalog["feature_input_order_drop_first"]
        else:
            raise Exception("Model type not in catalog")

        self._model = load_model(model_type, data.name, ext, cache, models_home, **kws)

        self._continuous = data.continous
        self._categoricals = data.categoricals

        # Preparing pipeline components
        self._use_pipeline = use_pipeline
        self._pipeline = self.__init_pipeline()

    def __init_pipeline(self):
        return [
            ("scaler", lambda x: scale(self.scaler, self._continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self._categoricals, x)),
            ("order", lambda x: order_data(self._feature_input_order, x)),
        ]

    def get_pipeline_element(self, key):
        """
        Returns a specific element of the pipeline

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    @property
    def pipeline(self):
        """
        Returns transformations steps for input before predictions.

        Returns
        -------
        pipeline : list
            List of (name, transform) tuples that are chained in the order in which they are preformed.
        """
        return self._pipeline

    def perform_pipeline(self, df):
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

        for trans_name, trans_function in self._pipeline:
            output = trans_function(output)

        return output

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
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        input = self.perform_pipeline(x) if self._use_pipeline else x

        if self._backend == "pytorch":
            # Pytorch model needs torch.Tensor as input
            if torch.is_tensor(input):
                device = "cuda" if input.is_cuda else "cpu"
                self._model = self._model.to(
                    device
                )  # Keep model and input on the same device
                return self._model(
                    input
                )  # If input is a tensor, the prediction will be a tensor too.
            else:
                # Convert ndArray input into torch tensor
                if isinstance(input, pd.DataFrame):
                    input = input.values
                input = torch.Tensor(input)

                self._model = self._model.to("cpu")
                output = self._model(input)

                # Convert output back to ndarray
                return output.detach().cpu().numpy()
        elif self._backend == "tensorflow":
            return self._model.predict(input)[:, 1].reshape(
                (-1, 1)
            )  # keep output in shape N x 1
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
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        input = self.perform_pipeline(x) if self._use_pipeline else x

        if self._backend == "pytorch":
            class_1 = 1 - self.predict(input)
            class_2 = self.predict(input)

            if torch.is_tensor(class_1):
                return torch.cat((class_1, class_2), dim=1)
            else:
                return np.array(list(zip(class_1, class_2))).reshape((-1, 2))

        elif self._backend == "tensorflow":
            return self._model.predict(input)
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    @property
    def use_pipeline(self):
        """
        Returns if the ML model uses the pipeline for predictions

        Returns
        -------
        bool
        """
        return self._use_pipeline

    @use_pipeline.setter
    def use_pipeline(self, use_pipe):
        """
        Sets if the ML model should use the pipeline before prediction.

        Parameters
        ----------
        use_pipe : bool
            If true, the model uses a transformation pipeline before prediction.

        Returns
        -------

        """
        self._use_pipeline = use_pipe
