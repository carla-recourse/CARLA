from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from carla.data.catalog import DataCatalog
from carla.data.load_catalog import load_catalog
from carla.models.api import MLModel
from carla.models.pipelining import encode, order_data, scale

from .load_model import load_model


class MLModelCatalog(MLModel):
    """
    Use pretrained classifier.

    Parameters
    ----------
    data : data.catalog.DataCatalog Class
        Correct dataset for ML model.
    model_type : {'ann', 'linear'}
        Architecture.
    backend : {'tensorflow', 'pytorch'}
        Specifies the used framework.
    cache : boolean, default: True
        If True, try to load from the local cache first, and save to the cache.
        if a download is required.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to the read model function
    use_pipeline : bool, default: False
        If true, the model uses a pipeline before predict and predict_proba to preprocess the input data.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model
    get_pipeline_element:
        Returns a specific element of the pipeline
    perform_pipeline:
        Transforms input for prediction into correct form.

    Returns
    -------
    None
    """

    def __init__(
        self,
        data: DataCatalog,
        model_type: str,
        backend: str = "tensorflow",
        cache: bool = True,
        models_home: str = None,
        use_pipeline: bool = False,
        **kws
    ) -> None:
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch" and "tensorflow".
        Possible models are corrently "ann".


        """
        self._backend = backend

        if self._backend == "pytorch":
            ext = "pt"
            encoding_method = "OneHot"
        elif self._backend == "tensorflow":
            ext = "h5"
            encoding_method = "OneHot_drop_binary"
        else:
            raise ValueError(
                "Backend not available, please choose between pytorch and tensorflow"
            )
        super().__init__(data, encoding_method=encoding_method)

        # Load catalog
        catalog_content = ["ann", "linear"]
        catalog = load_catalog("mlmodel_catalog.yaml", data.name, catalog_content)  # type: ignore

        if model_type not in catalog:
            raise ValueError("Model type not in model catalog")
        self._catalog = catalog[model_type][self._backend]
        self._feature_input_order = self._catalog["feature_order"]

        self._model = load_model(model_type, data.name, ext, cache, models_home, **kws)

        self._continuous = data.continous
        self._categoricals = data.categoricals

        # Preparing pipeline components
        self._use_pipeline = use_pipeline
        self._pipeline = self.__init_pipeline()

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self._continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self._categoricals, x)),
            ("order", lambda x: order_data(self._feature_input_order, x)),
        ]

    def get_pipeline_element(self, key: str) -> Callable:
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
    def pipeline(self) -> List[Tuple[str, Callable]]:
        """
        Returns transformations steps for input before predictions.

        Returns
        -------
        pipeline : list
            List of (name, transform) tuples that are chained in the order in which they are preformed.
        """
        return self._pipeline

    def perform_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
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
    def feature_input_order(self) -> List[str]:
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
    def backend(self) -> str:
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
    def raw_model(self) -> Any:
        """
        Returns the raw ml model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        return self._model

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
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
            return self.predict_proba(input)[:, 1].reshape((-1, 1))
        elif self._backend == "tensorflow":
            # keep output in shape N x 1
            return self._model.predict(input)[:, 1].reshape((-1, 1))
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
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
            # Keep model and input on the same device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)

            if isinstance(input, pd.DataFrame):
                input = input.values
            input, tensor_output = (
                (torch.Tensor(input), False)
                if not torch.is_tensor(input)
                else (input, True)
            )
            input = input.to(device)

            output = self._model(input)

            if tensor_output:
                return output
            else:
                return output.detach().cpu().numpy()

        elif self._backend == "tensorflow":
            return self._model.predict(input)
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    @property
    def use_pipeline(self) -> bool:
        """
        Returns if the ML model uses the pipeline for predictions

        Returns
        -------
        bool
        """
        return self._use_pipeline

    @use_pipeline.setter
    def use_pipeline(self, use_pipe: bool) -> None:
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
