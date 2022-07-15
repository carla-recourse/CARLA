from typing import Dict

import pandas as pd

from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.autoencoder import CSVAE
from carla.recourse_methods.catalog.crud.library import counterfactual_search
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)


class CRUD(RecourseMethod):
    """
    Implementation of CRUD [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Restriction
        * Currently working only with Pytorch models

    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "data_name": str
            name of the dataset
        * "lr": float, default: 0.008
            Learning rate for gradient descent.
        * "lambda_param": float, default: 0.001
            Weight factor for loss in counterfactual search.
        * "target_class" list, default: [0, 1]
            List of one-hot-encoded target class.
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "max_iter": int, default: 2000
            Number of search steps to find a counterfactual.
        * "optimizer": {RMSprop, Adam}
            Optimizer for counterfactual search.
        * "vae_params": Dict
            With parameter for VAE.
            + "layers": list
                List with number of neurons per layer, incl. input and latent dimension.
            + "train": bool, default: True
                Decides if a new Autoencoder will be learned.
            + "epochs": int, default: 5
                Number of epcchs to train VAE
            + "lr": float, default: 1e-3
                Learning rate for VAE training
            + "batch_size": int, default: 32
                Batch-size for VAE training

    .. [1] M. Downs, J. Chu, Yacoby Y, Doshi-Velez F, WeiWei P. CRUDS: Counterfactual Recourse Using Disentangled
            Subspaces. ICML Workshop on Human Interpretability in Machine Learning. 2020 :1-23.
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "target_class": [0, 1],
        "lambda_param": 0.001,
        "optimizer": "RMSprop",
        "lr": 0.008,
        "max_iter": 2000,
        "binary_cat_features": True,
        "vae_params": {
            "layers": None,
            "train": True,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    def __init__(self, mlmodel, hyperparams: Dict = None):

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self._target_class = checked_hyperparams["target_class"]
        self._lambda_param = checked_hyperparams["lambda_param"]
        self._optimizer = checked_hyperparams["optimizer"]
        self._lr = checked_hyperparams["lr"]
        self._max_iter = checked_hyperparams["max_iter"]
        self._binary_cat_features = checked_hyperparams["binary_cat_features"]

        vae_params = checked_hyperparams["vae_params"]
        self._csvae = CSVAE(
            checked_hyperparams["data_name"],
            vae_params["layers"],
            mlmodel.get_mutable_mask(),
        )

        if vae_params["train"]:
            self._csvae.fit(
                data=mlmodel.data.df[
                    mlmodel.feature_input_order + [mlmodel.data.target]
                ],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                self._csvae.load(self._mlmodel.data.df.shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

    def get_counterfactuals(self, factuals: pd.DataFrame):

        factuals = pd.concat(
            [
                self._mlmodel.get_ordered_features(factuals),
                factuals[self._mlmodel.data.target],
            ],
            axis=1,
        )

        # pay attention to categorical features
        encoded_feature_names = self._mlmodel.data.encoder.get_feature_names(
            self._mlmodel.data.categorical
        )
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

        df_cfs = factuals.apply(
            lambda x: counterfactual_search(
                self._mlmodel,
                self._csvae,
                x.reshape((1, -1)),
                cat_features_indices,
                self._binary_cat_features,
                self._target_class,
                self._lambda_param,
                self._optimizer,
                self._lr,
                self._max_iter,
            ),
            raw=True,
            axis=1,
        )

        cf_df = check_counterfactuals(
            self._mlmodel,
            df_cfs.drop(self._mlmodel.data.target, axis=1),
            factuals.index,
        )
        cf_df = self._mlmodel.get_ordered_features(cf_df)
        return cf_df
