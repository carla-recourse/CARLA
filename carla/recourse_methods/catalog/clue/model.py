import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from carla.models.pipelining import encode, scale
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.clue.library import (  # CLUE,; vae_gradient_search,
    VAE_gauss_cat_net,
    training,
)


class Clue(RecourseMethod):
    def __init__(self, data, mlmodel, hyperparams):
        """

        Parameters
        ----------
        data : data.api.Data
            Underlying dataset we want to build counterfactuals for.
        mlmodel : models.api.MLModel
            ML model to build counterfactuals for.
        hyperparams : dict
            Hyperparameter which are needed for CLUE to generate counterfactuals.
            Structure:
                {
                "data_name": str,   [Name of the dataset]
                "train_vae": bool,  [Decides whether to load or train a vae]
                "width": int,   [Structure for VAE]
                "depth": int,   [Structure for VAE]
                "latent_dim": int   [Structure for VAE]
                }
        """
        self._mlmodel = mlmodel
        self._train_vae = hyperparams["train_vae"]
        self._width = hyperparams["width"]
        self._depth = hyperparams["depth"]
        self._latent_dim = hyperparams["latent_dim"]
        self._input_dimension = len(self._mlmodel.feature_input_order)
        self._data_name = hyperparams["data_name"]

        # normalize and encode data and instance
        self._df_norm_enc_data = scale(mlmodel.scaler, data.continous, data.raw)
        self._df_norm_enc_data = encode(
            mlmodel.encoder, data.categoricals, self._df_norm_enc_data
        )
        self._df_norm_enc_data = self._df_norm_enc_data[mlmodel.feature_input_order]

        self._vae = self.load_vae()

    def load_vae(self):
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
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        if self._train_vae:
            self.train_vae(path)

        # Authors say: 'For automatic explainer generation'
        flat_vae_bools = False
        cuda = torch.cuda.is_available()
        vae = VAE_gauss_cat_net(
            self._input_dimension,
            self._width,
            self._depth,
            self._latent_dim,
            pred_sig=False,
            lr=1e-4,
            cuda=cuda,
            flatten=flat_vae_bools,
        )

        vae.load(os.path.join(path, "theta_best.dat"))

        return vae

    def train_vae(self, path):
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
        )

    def get_counterfactuals(self, factuals):
        pass
