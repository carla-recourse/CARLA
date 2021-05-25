from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import recourse as rs
from lime.lime_tabular import LimeTabularExplainer

from carla.recourse_methods.processing import encode_feature_names

from ...api import RecourseMethod


class ActionableRecourse(RecourseMethod):
    def __init__(
        self,
        mlmodel,
        hyperparams: Dict,
        coeffs: Optional[np.ndarray] = None,
        intercepts: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initializing Actionable Recourse

        Restrictions
        ------------
        -   Actionable Recourse (AR) supports only binary categorical features.
            See implementation at https://github.com/ustunb/actionable-recourse/blob/master/examples/ex_01_quickstart.ipynb
        -   AR is only defined on linear models. To make it work for arbitrary non-linear networks
            we need to find coefficients for every instance, for example with lime.

        Warning
        -------
        - AR does not always find a counterfactual example. The probability of finding one raises for a high size
          of flip set.

        Parameters
        ----------
        data : carla.data.Data()
            Dataset
        mlmodel : carla.model.MLModel()
            ML model
        hyperparams : dict
            Dictionary containing hyperparameters.
            {"fs_size": int (size of generated flipset, default 100)}
        coeffs : np.ndArray
            Coefficients
        intercepts: np.ndArray
            Intercepts
        """
        super().__init__(mlmodel)
        self._data = mlmodel.data

        # normalize and encode data
        self._norm_enc_data = self.encode_normalize_order_factuals(
            self._data.raw, with_target=True
        )

        # Get hyperparameter
        self._fs_size = (
            100 if "fs_size" not in hyperparams.keys() else hyperparams["fs_size"]
        )
        self._discretize_continuous = (
            False
            if "discretize" not in hyperparams.keys()
            else hyperparams["discretize"]
        )
        self._sample_around_instance = (
            True if "sample" not in hyperparams.keys() else hyperparams["sample"]
        )

        # Build ActionSet
        self._action_set = rs.ActionSet(
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

    def get_lime_coefficients(
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
        action_set = self._action_set

        factuals_enc_norm = self.encode_normalize_order_factuals(factuals)

        # Check if we need lime to build coefficients
        if (coeffs is None) and (intercepts is None):
            print("Start generating LIME coefficients")
            coeffs, intercepts = self.get_lime_coefficients(factuals_enc_norm)
            print("Finished generating LIME coefficients")

        # generate counterfactuals
        for index, row in factuals_enc_norm.iterrows():
            # asserts are essential for mypy typechecking
            assert coeffs is not None
            assert intercepts is not None
            factual_enc_norm = row.values
            coeff = coeffs[index]
            intercept = intercepts[index]

            # Align action set to coefficients
            action_set.set_alignment(coefficients=coeff)

            # Build AR flipset
            fs = rs.Flipset(
                x=factual_enc_norm,
                action_set=action_set,
                coefficients=coeff,
                intercept=intercept,
            )
            fs_pop = fs.populate(total_items=self._fs_size)

            # Get actions to flip predictions
            actions = fs_pop.actions

            # Default counterfactual value if no action flips the prediction
            target_shape = (1, factual_enc_norm.shape[0])
            empty = np.empty(target_shape)
            empty[:] = np.nan
            counterfactual = empty

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
                    counterfactual = candidate_cf
                    break

            cfs.append(counterfactual)

        # Convert output into correct format
        cfs = np.array(cfs).squeeze()
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        df_cfs[self._mlmodel.data.target] = np.argmax(
            self._mlmodel.predict_proba(cfs), axis=1
        )

        return df_cfs
