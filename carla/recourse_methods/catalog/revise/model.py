from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from carla import log
from carla.data.api import Data
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.autoencoder import (
    VAEDataset,
    VariationalAutoencoder,
    train_variational_autoencoder,
)
from carla.recourse_methods.processing.counterfactuals import (
    check_counterfactuals,
    merge_default_parameters,
    reconstruct_encoding_constraints,
)


class Revise(RecourseMethod):
    """
    Implementation of Revise from Joshi et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    data: carla.data.Data
        Dataset to perform on
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
            name of the dataset
        * "lambda": float, default: 0.5
            Decides how similar the counterfactual is to the factual
        * "optimizer": {"adam", "rmsprop"}
            Optimizer for generation of counterfactuals.
        * "lr": float, default: 0.1
            Learning rate for Revise.
        * "max_iter": int, default: 1000
            Number of iterations for Revise optimization.
        * "target_class": List, default: [0, 1]
            List of one-hot-encoded target class.
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "vae_params": Dict
            With parameter for VAE.

            + "layers": list
                Number of neurons and layer of autoencoder.
            + "train": bool
                Decides if a new Autoencoder will be learned.
            + "lambda_reg": flot
                Hyperparameter for variational autoencoder.
            + "epochs": int
                Number of epcchs to train VAE
            + "lr": float
                Learning rate for VAE training
            + "batch_size": int
                Batch-size for VAE training

    .. [1] Shalmali Joshi, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh.2019.
            Towards Realistic  Individual Recourse  and Actionable Explanations  in Black-BoxDecision Making Systems.
            arXiv preprint arXiv:1907.09615(2019).
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "lambda": 0.5,
        "optimizer": "adam",
        "lr": 0.1,
        "max_iter": 1000,
        "target_class": [0, 1],
        "binary_cat_features": True,
        "vae_params": {
            "layers": None,
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    def __init__(self, mlmodel: MLModel, data: Data, hyperparams: Dict) -> None:
        super().__init__(mlmodel)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._target_column = data.target
        self._lambda = self._params["lambda"]
        self._optimizer = self._params["optimizer"]
        self._lr = self._params["lr"]
        self._max_iter = self._params["max_iter"]
        self._target_class = self._params["target_class"]
        self._binary_cat_features = self._params["binary_cat_features"]

        df_enc_norm_data = self.encode_normalize_order_factuals(
            data.raw, with_target=True
        )

        vae_params = self._params["vae_params"]
        self.vae = VariationalAutoencoder(
            self._params["data_name"],
            vae_params["layers"],
        )

        if vae_params["train"]:
            self.vae = train_variational_autoencoder(
                self.vae,
                self._mlmodel.data,
                self._mlmodel.scaler,
                self._mlmodel.encoder,
                self._mlmodel.feature_input_order,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                self.vae.load(df_enc_norm_data.shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        df_enc_norm_fact = self.encode_normalize_order_factuals(
            factuals, with_target=True
        )

        # pay attention to categorical features
        encoded_feature_names = self._mlmodel.encoder.get_feature_names(
            self._mlmodel.data.categoricals
        )
        cat_features_indices = [
            df_enc_norm_fact.columns.get_loc(feature)
            for feature in encoded_feature_names
        ]

        list_cfs = self._counterfactual_optimization(
            cat_features_indices, device, df_enc_norm_fact
        )

        cf_df = check_counterfactuals(self._mlmodel, list_cfs)

        return cf_df

    def _counterfactual_optimization(self, cat_features_indices, device, df_fact):
        # prepare data for optimization steps
        test_loader = torch.utils.data.DataLoader(
            VAEDataset(df_fact.values), batch_size=1, shuffle=False
        )

        list_cfs = []
        for query_instance, _ in test_loader:

            target = torch.FloatTensor(self._target_class).to(device)
            target_prediction = np.argmax(np.array(self._target_class))

            z = self.vae.encode(query_instance)[0].clone().detach().requires_grad_(True)

            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], self._lr)
                # z.requires_grad = True
            else:
                optim = torch.optim.RMSprop([z], self._lr)

            candidate_counterfactuals = []  # all possible counterfactuals
            # distance of the possible counterfactuals from the intial value -
            # considering distance as the loss function (can even change it just the distance)
            candidate_distances = []
            all_loss = []

            for idx in range(self._max_iter):
                cf = self.vae.decode(z)[0]
                cf = reconstruct_encoding_constraints(
                    cf, cat_features_indices, self._params["binary_cat_features"]
                )
                output = self._mlmodel.predict_proba(cf)[0]
                _, predicted = torch.max(output, 0)

                z.requires_grad = True
                loss = self._compute_loss(cf, query_instance, target)
                all_loss.append(loss)

                if predicted == target_prediction:
                    candidate_counterfactuals.append(
                        cf.cpu().detach().numpy().squeeze(axis=0)
                    )
                    candidate_distances.append(loss.cpu().detach().numpy())

                loss.backward()
                optim.step()
                optim.zero_grad()
                cf.detach_()

            # Choose the nearest counterfactual
            if len(candidate_counterfactuals):
                log.info("Counterfactual found!")
                array_counterfactuals = np.array(candidate_counterfactuals)
                array_distances = np.array(candidate_distances)

                index = np.argmin(array_distances)
                list_cfs.append(array_counterfactuals[index])
            else:
                log.info("No counterfactual found")
                list_cfs.append(query_instance.cpu().detach().numpy().squeeze(axis=0))
        return list_cfs

    def _compute_loss(self, cf_initialize, query_instance, target):

        loss_function = nn.BCELoss()
        output = self._mlmodel.predict_proba(cf_initialize)[0]

        # classification loss
        loss1 = loss_function(output, target)
        # distance loss
        loss2 = torch.norm((cf_initialize - query_instance), 1)

        return loss1 + self._lambda * loss2
