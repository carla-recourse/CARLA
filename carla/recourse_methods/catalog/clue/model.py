import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from carla.data.api import Data
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.clue.library import (
    VAE_gauss_cat_net,
    training,
    vae_gradient_search,
)
from carla.recourse_methods.processing import check_counterfactuals
from carla.recourse_methods.processing.counterfactuals import merge_default_parameters


class Clue(RecourseMethod):
    """
    Implementation of CLUE from Antorán et.al. [1]_.
    CLUE needs an variational autoencoder to generate counterfactual examples.
    By setting the train_ae key to True in hyperparams, a Pytorch VAE will be trained.

    Parameters
    ----------
    data : data.api.Data
            Underlying dataset we want to build counterfactuals for.
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            Identifies the loaded or saved autoencoder model
        * "train_vae": bool
            Decides whether to load or train a vae
        * "width": int
            Structure for VAE
        * "depth": int
            Structure for VAE
        * "latent_dim": int
            Structure for VAE
        * "batch_size": int
            Structure for VAE
        * "epochs": int
            Structure for VAE
        * "lr": int
            Structure for VAE
        * "early_stop": int
            Structure for VAE

    .. [1] Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, and José Miguel Hernández-Lobato.
            Getting a CLUE: A Method for Explaining Uncertainty Estimates. In International Conference on
            Learning Representations (ICLR).
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "train_vae": True,
        "width": 10,
        "depth": 3,
        "latent_dim": 12,
        "batch_size": 64,
        "epochs": 1,
        "lr": 0.001,
        "early_stop": 10,
    }

    def __init__(self, data: Data, mlmodel: MLModel, hyperparams: Dict) -> None:
        super().__init__(mlmodel)

        # get hyperparameter
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self._vae_training = checked_hyperparams["train_vae"]
        self._width = checked_hyperparams["width"]
        self._depth = checked_hyperparams["depth"]
        self._latent_dim = checked_hyperparams["latent_dim"]
        self._data_name = checked_hyperparams["data_name"]
        self._batch_size = checked_hyperparams["batch_size"]
        self._epochs = checked_hyperparams["epochs"]
        self._lr = checked_hyperparams["lr"]
        self._early_stop = checked_hyperparams["early_stop"]
        self._continous = self._mlmodel.data.continous
        self._categorical = self._mlmodel.data.categoricals

        # get input dimension
        # indicate dimensions of inputs -- input_dim_vec: (if binary = 2; if continuous = 1)
        input_dims_continuous = list(np.repeat(1, len(self._mlmodel.data.continous)))
        input_dims_binary = list(np.repeat(2, len(self._mlmodel.data.categoricals)))
        self._input_dimension = input_dims_continuous + input_dims_binary

        # normalize and encode data
        self._df_norm_enc_data = self.encode_normalize_order_factuals(data.raw)

        # load autoencoder
        self._vae = self._load_vae()

    def _load_vae(self):
        # save_path
        path = os.environ.get(
            "CF_MODELS",
            os.path.join(
                "~",
                "carla",
                "models",
                "autoencoders",
                "clue",
                "fc_VAE_{}_models".format(self._data_name),
            ),
        )

        if not os.path.isfile(path) and not self._train_vae:
            raise ValueError(
                'No pre-trained VAE available. Please set "train_vae" to true in parameter "hyperparams" to train a VAE.'
            )

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        if self._vae_training:
            self._train_vae(path)

        # Authors say: 'For automatic explainer generation'
        flat_vae_bools = False
        cuda = torch.cuda.is_available()
        vae = VAE_gauss_cat_net(
            self._input_dimension,
            self._width,
            self._depth,
            self._latent_dim,
            pred_sig=False,
            lr=self._lr,
            cuda=cuda,
            flatten=flat_vae_bools,
        )

        vae.load(os.path.join(path, "theta_best.dat"))

        return vae

    def _train_vae(self, path):
        # training
        x_train, x_test = train_test_split(
            self._df_norm_enc_data.values, train_size=0.7
        )

        # Error message when training VAE using float 64: -> Change to: float 32
        # "Expected object of scalar type Float but got scalar type Double for argument #2 'mat1' in call to _th_addmm"
        x_train = np.float32(x_train)
        x_test = np.float32(x_test)

        training(
            x_train,
            x_test,
            self._input_dimension,
            path,
            self._width,
            self._depth,
            self._latent_dim,
            self._batch_size,
            self._epochs,
            self._lr,
            self._early_stop,
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        list_cfs = []

        # normalize and encode data and instance
        df_norm_enc_factual = self.encode_normalize_order_factuals(factuals)

        for index, row in df_norm_enc_factual.iterrows():
            counterfactual = vae_gradient_search(row.values, self._mlmodel, self._vae)
            list_cfs.append(counterfactual)

        # Convert output into correct format
        df_cfs = check_counterfactuals(self._mlmodel, list_cfs)

        return df_cfs
