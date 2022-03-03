from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split

from carla.data.catalog import DataCatalog
from carla.data.causal_model.synthethic_data import ScmDataset
from carla.data.load_catalog import load
from carla.models.api import MLModel
from carla.models.pipelining import decode, descale, encode, order_data, scale

from .load_model import load_online_model, load_trained_model, save_model
from .train_model import train_model


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
    load_online: bool, default: True
        If true, a pretrained model is loaded. If false, a model is trained.

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
        load_online: bool = True,
        **kws,
    ) -> None:
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch" and "tensorflow".
        Possible models are corrently "ann" and "linear".


        """
        self._model_type = model_type
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

        if not isinstance(data, ScmDataset) and data.name != "custom":
            # Load catalog
            catalog_content = ["ann", "linear"]
            catalog = load("mlmodel_catalog.yaml", data.name, catalog_content)  # type: ignore

            if model_type not in catalog:
                raise ValueError("Model type not in model catalog")
            self._catalog = catalog[model_type][self._backend]
            self._feature_input_order = self._catalog["feature_order"]
        else:
            self._catalog = None
            encoded_features = list(self.encoder.get_feature_names(data.categorical))
            self._feature_input_order = list(
                np.sort(data.continuous + encoded_features)
            )

        self._continuous = data.continuous
        self._categorical = data.categorical

        # Preparing pipeline components
        self._use_pipeline = use_pipeline
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        if load_online:
            self._model = load_online_model(
                model_type, data.name, ext, cache, models_home, **kws
            )

    def _test_accuracy(self):
        data_df = self.data.raw
        x = data_df[list(set(data_df.columns) - {self.data.target})]
        y = data_df[self.data.target]
        x_train, x_test, y_train, y_test = train_test_split(x, y)

        prediction = (self.predict(x_test) > 0.5).flatten()
        correct = prediction == y_test
        print(f"approx. acc for model: {correct.mean()}")

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self._continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self._categorical, x)),
            ("order", lambda x: order_data(self._feature_input_order, x)),
        ]

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("encoder", lambda x: decode(self.encoder, self._categorical, x)),
            ("scaler", lambda x: descale(self.scaler, self._continuous, x)),
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

    @property
    def inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        """
        Returns transformations steps for output after predictions.

        Returns
        -------
        pipeline : list
            List of (name, transform) tuples that are chained in the order in which they are preformed.
        """
        return self._inverse_pipeline

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

    def perform_inverse_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms output after prediction back into original form.
        Only possible for DataFrames with preprocessing steps.

        Parameters
        ----------
        df : pd.DataFrame
            Contains normalized and encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction output denormalized and decoded

        """
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
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
    def model_type(self) -> str:
        """
        Describes the model type

        E.g., ann, linear

        Returns
        -------
        backend : str
            model type
        """
        return self._model_type

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

        if self._backend == "pytorch":
            input = x
            return self.predict_proba(input)[:, 1].reshape((-1, 1))
        elif self._backend == "tensorflow":
            # keep output in shape N x 1
            input = self.perform_pipeline(x) if self._use_pipeline else x
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

    def train(
        self,
        learning_rate,
        epochs,
        batch_size,
        force_train=False,
        hidden_size=[18, 9, 3],
    ):
        """

        Parameters
        ----------
        learning_rate: float
            Learning rate for the training.
        epochs: int
            Number of epochs to train for.
        batch_size: int
            Number of samples in each batch
        force_train: bool
            Force training, even if model already exists in cache.
        hidden_size: list[int]
            hidden_size[i] contains the number of nodes in layer [i]

        Returns
        -------

        """
        layer_string = "_".join([str(size) for size in hidden_size])
        if self.model_type == "linear":
            save_name = f"{self.model_type}"
        elif self.model_type == "ann":
            save_name = f"{self.model_type}_layers_{layer_string}"
        else:
            raise NotImplementedError("Model type not supported:", self.model_type)

        # try to load the model from disk, if that fails train the model instead.
        self._model = None
        if not force_train:
            self._model = load_trained_model(
                save_name=save_name, data_name=self.data.name, backend=self.backend
            )

            # sanity check to see if loaded model accuracy makes sense
            if self._model is not None:
                self._test_accuracy()

        if self._model is None or force_train:
            # preprocess data
            df_train = self.data.train_raw
            df_test = self.data.test_raw
            if self.use_pipeline:
                x_train = self.perform_pipeline(df_train)
                x_test = self.perform_pipeline(df_test)
            else:
                x_train = df_train[list(set(df_train.columns) - {self.data.target})]
                x_test = df_test[list(set(df_test.columns) - {self.data.target})]
            y_train = df_train[self.data.target]
            y_test = df_test[self.data.target]

            self._model = train_model(
                self,
                x_train,
                y_train,
                x_test,
                y_test,
                learning_rate,
                epochs,
                batch_size,
                hidden_size,
            )

            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.data.name,
                backend=self.backend,
            )
