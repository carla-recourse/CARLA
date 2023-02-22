from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA

from carla import log
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.autoencoder import VariationalAutoencoder
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
    reconstruct_encoding_constraints,
)


class CCHVAE(RecourseMethod):
    """
    Implementation of CCHVAE [1]_

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            name of the dataset
        * "n_search_samples": int, default: 300
            Number of generated candidate counterfactuals.
        * "p_norm": {1, 2}
            Defines L_p norm for distance calculation.
        * "step": float, default: 0.1
            Step size for each generated candidate counterfactual.
        * "max_iter": int, default: 1000
            Number of iterations per factual instance.
        * "clamp": bool, default: True
            Feature values will be clamped between 0 and 1
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "vae_params": Dict
            With parameter for VAE.

            + "layers": list
                List with number of neurons per layer.
            + "train": bool, default: True
                Decides if a new Autoencoder will be learned.
            + "lambda_reg": float, default: 1e-6
                Hyperparameter for variational autoencoder.
            + "epochs": int, default: 5
                Number of epochs to train VAE
            + "lr": float, default: 1e-3
                Learning rate for VAE training
            + "batch_size": int, default: 32
                Batch-size for VAE training

    .. [1] Pawelczyk, Martin, Klaus Broelemann and Gjergji Kasneci. “Learning Model-Agnostic Counterfactual Explanations
          for Tabular Data.” Proceedings of The Web Conference 2020 (2020): n. pag..
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "n_search_samples": 300,
        "p_norm": 1,
        "step": 0.1,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": True,
        "vae_params": {
            "layers": None,
            "train": True,
            "kl_weight": 0.3,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None) -> None:

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._n_search_samples = self._params["n_search_samples"]
        self._p_norm = self._params["p_norm"]
        self._step = self._params["step"]
        self._max_iter = self._params["max_iter"]
        self._clamp = self._params["clamp"]

        vae_params = self._params["vae_params"]
        self._generative_model = self._load_vae(
            self._mlmodel.data.df, vae_params, self._mlmodel, self._params["data_name"]
        )

    def _load_vae(
        self, data: pd.DataFrame, vae_params: Dict, mlmodel: MLModel, data_name: str
    ) -> VariationalAutoencoder:

        generative_model = VariationalAutoencoder(
            data_name, vae_params["layers"], mlmodel.get_mutable_mask()
        )

        if vae_params["train"]:
            generative_model.fit(
                xtrain=data[mlmodel.feature_input_order],
                kl_weight=vae_params["kl_weight"],
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                generative_model.load(vae_params["layers"][0])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

        return generative_model

    def _hyper_sphere_coordindates(
        self, instance, high: int, low: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param n_search_samples: int > 0
        :param instance: numpy input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :param p: float>= 1; norm
        :return: candidate counterfactuals & distances
        """
        delta_instance = np.random.randn(self._n_search_samples, instance.shape[1])
        dist = (
            np.random.rand(self._n_search_samples) * (high - low) + low
        )  # length range [l, h)
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
        return candidate_counterfactuals, dist

    def _counterfactual_search(
        self, step: int, factual: torch.Tensor, cat_features_indices: List
    ) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # init step size for growing the sphere
        low = 0
        high = step
        # counter
        count = 0
        counter_step = 1

        torch_fact = torch.from_numpy(factual).to(device)

        # get predicted label of instance
        instance_label = np.argmax(
            self._mlmodel.predict_proba(torch_fact.float()).cpu().detach().numpy(),
            axis=1,
        )

        # vectorize z
        z = self._generative_model.encode(
            torch_fact[:, self._generative_model.mutable_mask].float()
        )[0]
        # add the immutable features to the latents
        z = torch.cat([z, torch_fact[:, ~self._generative_model.mutable_mask]], dim=-1)
        z = z.cpu().detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self._n_search_samples, axis=0)

        # make copy such that we later easily combine the immutables and the reconstructed mutables
        fact_rep = torch_fact.reshape(1, -1).repeat_interleave(
            self._n_search_samples, dim=0
        )

        candidate_dist: List = []
        x_ce: Union[np.ndarray, torch.Tensor] = np.array([])
        while count <= self._max_iter or len(candidate_dist) <= 0:
            count = count + counter_step
            if count > self._max_iter:
                log.debug("No counterfactual example found")
                return x_ce[0]

            # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
            latent_neighbourhood, _ = self._hyper_sphere_coordindates(z_rep, high, low)
            torch_latent_neighbourhood = (
                torch.from_numpy(latent_neighbourhood).to(device).float()
            )
            x_ce = self._generative_model.decode(torch_latent_neighbourhood)

            # add the immutable features to the reconstruction
            temp = fact_rep.clone()
            temp[:, self._generative_model.mutable_mask] = x_ce.double()
            x_ce = temp

            x_ce = reconstruct_encoding_constraints(
                x_ce, cat_features_indices, self._params["binary_cat_features"]
            )
            x_ce = x_ce.detach().cpu().numpy()
            x_ce = x_ce.clip(0, 1) if self._clamp else x_ce

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self._p_norm == 1:
                distances = np.abs((x_ce - torch_fact.cpu().detach().numpy())).sum(
                    axis=1
                )
            elif self._p_norm == 2:
                distances = LA.norm(x_ce - torch_fact.cpu().detach().numpy(), axis=1)
            else:
                raise ValueError("Possible values for p_norm are 1 or 2")

            # counterfactual labels
            y_candidate = np.argmax(
                self._mlmodel.predict_proba(torch.from_numpy(x_ce).float())
                .cpu()
                .detach()
                .numpy(),
                axis=1,
            )
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals = x_ce[indices]
            candidate_dist = distances[indices]
            # no candidate found & push search range outside
            if len(candidate_dist) == 0:
                low = high
                high = low + step
            elif len(candidate_dist) > 0:
                # certain candidates generated
                min_index = np.argmin(candidate_dist)
                log.debug("Counterfactual example found")
                return candidate_counterfactuals[min_index]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = self._mlmodel.get_ordered_features(factuals)

        encoded_feature_names = self._mlmodel.data.encoder.get_feature_names(
            self._mlmodel.data.categorical
        )
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

        df_cfs = factuals.apply(
            lambda x: self._counterfactual_search(
                self._step, x.reshape((1, -1)), cat_features_indices
            ),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
