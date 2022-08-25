from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from carla import log
from carla.recourse_methods.catalog.roar.library import roar_recourse
from carla.recourse_methods.processing import check_counterfactuals

from ...api import RecourseMethod
from ...processing.counterfactuals import merge_default_parameters


class Roar(RecourseMethod):
    """
    Implementation of ROAR [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.
    coeffs : np.ndArray, optional
        Coefficients. Will be approximated by LIME if None
    intercepts: np.ndArray, optional
        Intercepts. Will be approximated by LIME if None

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "feature_cost": str, optional
            List with costs per feature.
        * "lr": float, default: 0.01
            Learning rate for gradient descent.
        * "lambda_": float, default: 0.01
            Weight factor for feature_cost.
        * "delta_max": float, default: 0.01
            Maximum perturbation for weights
        * "t_max_min": float, default: 0.5
            Maximum time of search.
        * "norm": int, default: 1
            L-norm to calculate cost.
        * "loss_type": {"MSE", "BCE"}
            String for loss function.
        * "y_target" list, default: [0, 1]
            List of one-hot-encoded target class.
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "discretize": bool, default: False
            Parameter for LIME sampling.
        * "sample": bool, default: True
            LIME sampling around instance.
        * "loss_threshold": float, default: 0.001
            Threshold for loss difference
        * "lime_seed": int, default: 0
            Seed when generating LIME coefficients
        * "seed": int, default: 0
            Seed for torch when calculating counterfactuals

    - Restrictions
        *   ROAR is only defined on linear models. To make it work for arbitrary non-linear networks
            we need to find coefficients for every instance, for example with lime.
        *   Currently working only with Pytorch models.
    - Warning
        *   Not guaranteed to find recourse.

    .. [1] Upadhyay, S., Joshi, S., & Lakkaraju, H. (2021). Towards Robust and Reliable Algorithmic Recourse. NeurIPS.
    """

    _DEFAULT_HYPERPARAMS = {
        "feature_cost": "_optional_",
        "lr": 0.01,
        "lambda_": 0.01,
        "delta_max": 0.01,
        "norm": 1,
        "t_max_min": 0.5,
        "loss_type": "BCE",
        "y_target": [0, 1],
        "binary_cat_features": True,
        "loss_threshold": 1e-3,
        "discretize": False,
        "sample": True,
        "lime_seed": 0,
        "seed": 0,
    }

    def __init__(
        self,
        mlmodel,
        hyperparams: Dict,
        coeffs: Optional[np.ndarray] = None,
        intercepts: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(mlmodel)

        self._data = mlmodel.data
        self._mlmodel = mlmodel
        self._coeffs = coeffs
        self._intercepts = intercepts

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self._feature_costs = checked_hyperparams["feature_cost"]
        self._lr = checked_hyperparams["lr"]
        self._lambda_param = checked_hyperparams["lambda_"]
        self._delta_max = checked_hyperparams["delta_max"]
        self._norm = checked_hyperparams["norm"]
        self._t_max_min = checked_hyperparams["t_max_min"]
        self._loss_type = checked_hyperparams["loss_type"]
        self._y_target = checked_hyperparams["y_target"]
        self._binary_cat_features = checked_hyperparams["binary_cat_features"]
        self._loss_threshold = checked_hyperparams["loss_threshold"]
        self._discretize_continuous = checked_hyperparams["discretize"]
        self._sample_around_instance = checked_hyperparams["sample"]
        self._lime_seed = checked_hyperparams["lime_seed"]
        self._seed = checked_hyperparams["seed"]

    def _get_lime_coefficients(
        self, factuals: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ROAR Recourse is only defined on linear models. To make it work for arbitrary non-linear networks
        we need to find the lime coefficients for every instance.

        Parameters
        ----------
        factuals : pd.DataFrame
            Instances we want to get lime coefficients

        Returns
        -------
        coeffs : np.ndArray
        intercepts : np.ndArray

        """

        np.random.seed(self._lime_seed)

        coeffs = np.zeros(factuals.shape)
        intercepts = []
        lime_data = self._data.df[self._mlmodel.feature_input_order]
        lime_label = self._data.df[self._data.target]

        lime_exp = LimeTabularExplainer(
            training_data=lime_data.values,
            training_labels=lime_label,
            feature_names=self._mlmodel.feature_input_order,
            discretize_continuous=self._discretize_continuous,
            sample_around_instance=self._sample_around_instance,
            categorical_names=[
                cat
                for cat in self._mlmodel.feature_input_order
                if cat not in self._data.continuous
            ]
            # self._data.encoded_normalized's categorical features contain feature name and value, separated by '_'
            # while self._data.categorical do not contain those additional values.
        )

        for index, row in factuals.iterrows():
            factual = row.values
            explanations = lime_exp.explain_instance(
                factual,
                self._mlmodel.predict_proba,
                num_features=len(self._mlmodel.feature_input_order),
            )
            intercepts.append(explanations.intercept[1])

            for tpl in explanations.local_exp[1]:
                coeffs[index][tpl[0]] = tpl[1]

        return coeffs, np.array(intercepts)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.reset_index()
        factuals = self._mlmodel.get_ordered_features(factuals)

        encoded_feature_names = self._mlmodel.data.encoder.get_feature_names(
            self._mlmodel.data.categorical
        )
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

        coeffs = self._coeffs
        intercepts = self._intercepts

        # Calculate coefficients and intercept (if not given) and reshape to match the shape that LIME generates
        # If Model linear then extract coefficients and intercepts from raw model directly
        # If Molel ANN then use LIME to generate the coefficients
        if (coeffs is None) or (intercepts is None):
            if self._mlmodel.model_type == "linear":
                coeffs_neg = (
                    self._mlmodel.raw_model.output.weight.cpu().detach()[0].numpy()
                )
                coeffs_pos = (
                    self._mlmodel.raw_model.output.weight.cpu().detach()[1].numpy()
                )

                intercepts_neg = np.array(
                    self._mlmodel.raw_model.output.bias.cpu().detach()[0].numpy()
                )
                intercepts_pos = np.array(
                    self._mlmodel.raw_model.output.bias.cpu().detach()[1].numpy()
                )

                self._coeffs = coeffs_pos - coeffs_neg
                self._intercepts = intercepts_pos - intercepts_neg

                # Local explanations via LIME generate coeffs and intercepts per instance, while global explanations
                # via input parameter need to be set into correct shape [num_of_instances, num_of_features]
                coeffs = np.vstack([self._coeffs] * factuals.shape[0])
                intercepts = np.vstack([self._intercepts] * factuals.shape[0]).squeeze(
                    axis=1
                )
            elif self._mlmodel.model_type == "ann":
                log.info("Start generating LIME coefficients")
                coeffs, intercepts = self._get_lime_coefficients(factuals)
                log.info("Finished generating LIME coefficients")
            else:
                raise ValueError(
                    f"Model type {self._mlmodel.model_type} not supported in ROAR recourse method"
                )
        else:
            # Coeffs and intercepts should be numpy arrays of shape (num_features,) and () respectively
            if (len(coeffs.shape) != 1) or (coeffs.shape[0] != factuals.shape[1]):
                raise ValueError(
                    "Incorrect shape of coefficients. Expected shape: (num_features,)"
                )
            if len(intercepts.shape) != 0:
                raise ValueError("Incorrect shape of coefficients. Expected shape: ()")

            # Reshape to desired shape: (num_of_instances, num_of_features)
            coeffs = np.vstack([self._coeffs] * factuals.shape[0])
            intercepts = np.vstack([self._intercepts] * factuals.shape[0]).squeeze(
                axis=1
            )

        cfs = []
        for index, row in factuals.iterrows():

            coeff = coeffs[index]
            intercept = intercepts[index]

            counterfactual = roar_recourse(
                self._mlmodel.raw_model,
                row.to_numpy().reshape((1, -1)),
                coeff,
                intercept,
                cat_features_indices,
                binary_cat_features=self._binary_cat_features,
                feature_costs=self._feature_costs,
                lr=self._lr,
                lambda_param=self._lambda_param,
                delta_max=self._delta_max,
                y_target=self._y_target,
                norm=self._norm,
                t_max_min=self._t_max_min,
                loss_type=self._loss_type,
                loss_threshold=self._loss_threshold,
                seed=self._seed,
            )
            cfs.append(counterfactual)

        # Convert output into correct format
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)

        return df_cfs
