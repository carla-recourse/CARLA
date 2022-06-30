from typing import Any, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from carla.data.catalog.online_catalog import DataCatalog, OnlineCatalog
from carla.data.load_catalog import load
from carla.models.api import MLModel

from .load_model import load_online_model, load_trained_model, save_model
from .train_model import train_model


class MLModelCatalog(MLModel):
    """
    Use pretrained classifier.

    Parameters
    ----------
    data : data.catalog.DataCatalog Class
        Correct dataset for ML model.
    model_type : {'ann', 'linear', 'forest'}
        The model architecture. Artificial Neural Network, Logistic Regression, and Random Forest respectively.
    backend : {'tensorflow', 'pytorch', 'sklearn', 'xgboost'}
        Specifies the used framework. Tensorflow and PyTorch only support 'ann' and 'linear'. Sklearn and Xgboost only support 'forest'.
    cache : boolean, default: True
        If True, try to load from the local cache first, and save to the cache if a download is required.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to the read model function
    load_online: bool, default: True
        If true, a pretrained model is loaded. If false, a model is trained.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model

    Returns
    -------
    None
    """

    def __init__(
        self,
        data: DataCatalog,
        model_type: str,
        backend: str,
        cache: bool = True,
        models_home: str = None,
        load_online: bool = True,
        **kws,
    ) -> None:
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch", "tensorflow" for "ann" and "linear" models.
        Possible backends are currently "sklearn", "xgboost" for "forest" models.

        """
        self._model_type = model_type
        self._backend = backend
        self._continuous = data.continuous
        self._categorical = data.categorical

        if self._backend == "pytorch":
            ext = "pt"
        elif self._backend == "tensorflow":
            ext = "h5"
        elif self._backend == "sklearn":
            ext = "skjoblib"
        elif self._backend == "xgboost":
            ext = "xgjoblib"
        else:
            raise ValueError(
                'Backend not available, please choose between "pytorch", "tensorflow", "sklearn", or "xgboost".'
            )
        super().__init__(data)

        # Only datasets in our catalog have a saved yaml file
        if isinstance(data, OnlineCatalog):
            # Load catalog
            catalog_content = ["ann", "linear"]
            catalog = load("mlmodel_catalog.yaml", data.name, catalog_content)  # type: ignore

            if model_type not in catalog:
                raise ValueError("Model type not in model catalog")

            self._catalog = catalog[model_type][self._backend]
            self._feature_input_order = self._catalog["feature_order"]
        else:
            if data._identity_encoding:
                encoded_features = data.categorical
            else:
                encoded_features = list(
                    data.encoder.get_feature_names(data.categorical)
                )

            self._catalog = None

            if model_type == "forest":
                self._feature_input_order = list(np.sort(data.continuous))
            else:
                self._feature_input_order = list(
                    np.sort(data.continuous + encoded_features)
                )

        if load_online:
            self._model = load_online_model(
                model_type, data.name, ext, cache, models_home, **kws
            )

    def _test_accuracy(self):
        # get preprocessed data
        df_test = self.data.df_test

        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        prediction = (self.predict(x_test) > 0.5).flatten()
        correct = prediction == y_test
        print(f"test accuracy for model: {correct.mean()}")

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
        Returns the raw ML model built on its framework

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
            raise ValueError(
                "Input shape has to be two-dimensional, (instances, features)."
            )

        if self._backend == "pytorch":
            return self.predict_proba(x)[:, 1].reshape((-1, 1))
        elif self._backend == "tensorflow":
            # keep output in shape N x 1
            # order data (column-wise) before prediction
            x = self.get_ordered_features(x)
            return self._model.predict(x)[:, 1].reshape((-1, 1))
        elif self._backend == "sklearn" or self._backend == "xgboost":
            return self._model.predict(self.get_ordered_features(x))
        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
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

        # order data (column-wise) before prediction
        x = self.get_ordered_features(x)

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        if self._backend == "pytorch":

            # Keep model and input on the same device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)

            if isinstance(x, pd.DataFrame):
                _x = x.values
            elif isinstance(x, torch.Tensor):
                _x = x.clone()
            else:
                _x = x.copy()

            # If the input was a tensor, return a tensor. Else return a np array.
            tensor_output = torch.is_tensor(x)
            if not tensor_output:
                _x = torch.Tensor(_x)

            # input, tensor_output = (
            #     (torch.Tensor(x), False) if not torch.is_tensor(x) else (x, True)
            # )

            _x = _x.to(device)
            output = self._model(_x)

            if tensor_output:
                return output
            else:
                return output.detach().cpu().numpy()

        elif self._backend == "tensorflow":
            return self._model.predict(x)
        elif self._backend == "sklearn" or self._backend == "xgboost":
            return self._model.predict_proba(x)
        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    @property
    def tree_iterator(self):
        """
        A method needed specifically for tree methods. This method should return a list of individual trees that make up the forest.

        Returns
        -------

        """
        if self.model_type != "forest":
            return None
        elif self.backend == "sklearn":
            return self._model
        elif self.backend == "xgboost":
            # make a copy of the trees, else feature names are not saved
            booster_it = [booster for booster in self.raw_model.get_booster()]
            # set the feature names
            for booster in booster_it:
                booster.feature_names = self.feature_input_order
            return booster_it

    def train(
        self,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        force_train=False,
        hidden_size=[18, 9, 3],
        n_estimators=5,
        max_depth=5,
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
        n_estimators: int
            Number of estimators in forest.
        max_depth: int
            Max depth of trees in the forest.

        Returns
        -------

        """
        layer_string = "_".join([str(size) for size in hidden_size])
        if self.model_type == "linear" or self.model_type == "forest":
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

        # if model loading failed or force_train flag set to true.
        if self._model is None or force_train:
            # get preprocessed data
            df_train = self.data.df_train
            df_test = self.data.df_test

            x_train = df_train[list(set(df_train.columns) - {self.data.target})]
            y_train = df_train[self.data.target]
            x_test = df_test[list(set(df_test.columns) - {self.data.target})]
            y_test = df_test[self.data.target]

            # order data (column-wise) before training
            x_train = self.get_ordered_features(x_train)
            x_test = self.get_ordered_features(x_test)

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
                n_estimators,
                max_depth,
            )

            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.data.name,
                backend=self.backend,
            )
