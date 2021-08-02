from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import recourse as rs
from lime.lime_tabular import LimeTabularExplainer

from carla import log
from carla.recourse_methods.processing import encode_feature_names

from ...api import RecourseMethod
from ...processing.counterfactuals import merge_default_parameters


class ActionableRecourse(RecourseMethod):
    """
    Implementation of Actionable Recourse from Ustun et.al. [1]_

    Parameters
    ----------
    data : carla.data.Data
        Dataset
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
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "fs_size": int, default: 100
            Size of generated flipset.
        * "discretize": bool, default: False
            Parameter for LIME sampling.
        * "sample": boo, default: True
            Lime sampling around instance.
    - Restrictions
        *   Actionable Recourse (AR) supports only binary categorical features.
            See implementation at https://github.com/ustunb/actionable-recourse/blob/master/examples/ex_01_quickstart.ipynb
        *   AR is only defined on linear models. To make it work for arbitrary non-linear networks
            we need to find Actionab coefficients for every instance, for example with lime.
    - Warning
        *   AR does not always find a counterfactual example. The probability of finding one raises for a high size
            of flip set.

    .. [1] Berk Ustun, Alexander Spangher, and Y. Liu. 2019. Actionable Recourse in Linear Classification.
        InProceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)
    """

    _DEFAULT_HYPERPARAMS = {
        "fs_size": 100,
        "discretize": False,
        "sample": True,
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

        # normalize and encode data
        self._norm_enc_data = self.encode_normalize_order_factuals(
            self._data.raw, with_target=True
        )

        # Get hyperparameter
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self._fs_size = checked_hyperparams["fs_size"]
        self._discretize_continuous = checked_hyperparams["discretize"]
        self._sample_around_instance = checked_hyperparams["sample"]

        # Build ActionSet
        self.action_set = rs.ActionSet(
            X=self._norm_enc_data[self._mlmodel.feature_input_order]
        )

        # transform immutable feature names into encoded feature names of self._data.encoded_normalized
        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

        for feature in self._immutables:
            self._action_set[feature].mutable = False
            self._action_set[feature].actionable = False

        self._coeffs, self._intercepts = coeffs, intercepts

    @property
    def action_set(self):
        """
        Contains dictionary with possible actions for every input feature.

        Returns
        -------
        dict
        """
        return self._action_set

    @action_set.setter
    def action_set(self, act_set):
        self._action_set = act_set

    def _get_lime_coefficients(
        self, factuals: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Actionable Recourse is only defined on linear models. To make it work for arbitrary non-linear networks
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
        coeffs = np.zeros(factuals.shape)
        intercepts = []
        lime_data = self._norm_enc_data[self._mlmodel.feature_input_order]
        lime_label = self._norm_enc_data[self._data.target]

        lime_exp = LimeTabularExplainer(
            training_data=lime_data.values,
            training_labels=lime_label,
            feature_names=self._mlmodel.feature_input_order,
            discretize_continuous=self._discretize_continuous,
            sample_around_instance=self._sample_around_instance,
            categorical_names=[
                cat
                for cat in self._mlmodel.feature_input_order
                if cat not in self._data.continous
            ]
            # self._data.encoded_normalized's categorical features contain feature name and value, separated by '_'
            # while self._data.categoricals do not contain those additional values.
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
        cfs = []
        coeffs = self._coeffs
        intercepts = self._intercepts
        action_set = self.action_set

        # to keep matching indexes for iterrows and coeffs
        factuals = factuals.reset_index()
        factuals_enc_norm = self.encode_normalize_order_factuals(factuals)

        # Check if we need lime to build coefficients
        if (coeffs is None) and (intercepts is None):
            log.info("Start generating LIME coefficients")
            coeffs, intercepts = self._get_lime_coefficients(factuals_enc_norm)
            log.info("Finished generating LIME coefficients")
        else:
            # Local explanations via LIME generate coeffs and intercepts per instance, while global explanations
            # via input parameter need to be set into correct shape [num_of_instances, num_of_features]
            coeffs = np.vstack([self._coeffs] * factuals.shape[0])
            intercepts = np.vstack([self._intercepts] * factuals.shape[0]).squeeze(
                axis=1
            )

        # generate counterfactuals
        for index, row in factuals_enc_norm.iterrows():
            # asserts are essential for mypy typechecking
            assert coeffs is not None
            assert intercepts is not None
            factual_enc_norm = row.values
            coeff = coeffs[index]
            intercept = intercepts[index]

            # Default counterfactual value if no action flips the prediction
            target_shape = factual_enc_norm.shape[0]
            empty = np.empty(target_shape)
            empty[:] = np.nan
            counterfactual = empty

            # Align action set to coefficients
            action_set.set_alignment(coefficients=coeff)

            # Build AR flipset
            fs = rs.Flipset(
                x=factual_enc_norm,
                action_set=action_set,
                coefficients=coeff,
                intercept=intercept,
            )
            try:
                fs_pop = fs.populate(total_items=self._fs_size)
            except (ValueError, KeyError):
                log.warning(
                    "Actionable Recourse is not able to produce a counterfactual explanation for instance {}".format(
                        index
                    )
                )
                log.warning(row.values)
                cfs.append(counterfactual)
                continue

            # Get actions to flip predictions
            actions = fs_pop.actions

            for action in actions:
                candidate_cf = (factual_enc_norm + action).reshape(
                    (1, -1)
                )  # Reshape to keep two-dim. input
                # Check if candidate counterfactual really flipps the prediction of ML model
                pred_cf = np.argmax(self._mlmodel.predict_proba(candidate_cf))
                pred_f = np.argmax(
                    self._mlmodel.predict_proba(factual_enc_norm.reshape((1, -1)))
                )
                if pred_cf != pred_f:
                    counterfactual = candidate_cf.squeeze()
                    break

            cfs.append(counterfactual)

        # Convert output into correct format
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        df_cfs[self._mlmodel.data.target] = np.argmax(
            self._mlmodel.predict_proba(cfs), axis=1
        )

        return df_cfs
