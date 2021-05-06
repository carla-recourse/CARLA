import numpy as np
import pandas as pd
import recourse as rs
from lime.lime_tabular import LimeTabularExplainer

from carla.models.pipelining import encode, scale
from carla.recourse_methods.processing import encoded_immutables

from ...api import RecourseMethod


class ActionableRecourse(RecourseMethod):
    def __init__(self, mlmodel, hyperparams, coeffs=None, intercepts=None):
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
        intercepts
        """
        self._data = mlmodel.data
        self._mlmodel = mlmodel

        # normalize and encode data
        self._norm_enc_data = scale(
            mlmodel.scaler, self._data.continous, self._data.raw
        )
        self._norm_enc_data = encode(
            mlmodel.encoder, self._data.categoricals, self._norm_enc_data
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
        self._immutables = encoded_immutables(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

        for feature in self._immutables:
            self._action_set[feature].mutable = False
            self._action_set[feature].actionable = False

        self._coeffs, self._intercepts = coeffs, intercepts

    def get_lime_coefficients(self, factuals):
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

        for i in range(factuals.shape[0]):
            factual = factuals.iloc[i].T.values
            explanations = lime_exp.explain_instance(
                factual,
                self._mlmodel.predict_proba,
                num_features=len(self._mlmodel.feature_input_order),
            )
            intercepts.append(explanations.intercept[1])

            for tpl in explanations.local_exp[1]:
                coeffs[i][tpl[0]] = tpl[1]

        return coeffs, intercepts

    def get_counterfactuals(self, factuals):
        cfs = []

        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        if querry_instances.shape[0] == 0:
            raise ValueError("Factuals should not be empty")

        # preprocessing for lime
        scaler = self._mlmodel.scaler
        encoder = self._mlmodel.encoder
        querry_instances[self._data.continous] = scaler.transform(
            querry_instances[self._data.continous]
        )  # normalize
        encoded_features = encoder.get_feature_names(self._data.categoricals)
        querry_instances[encoded_features] = encoder.transform(
            querry_instances[self._data.categoricals]
        )  # encode
        factuals_enc_norm = querry_instances[
            self._mlmodel.feature_input_order
        ]  # get feature order

        # Check if we need lime to build coefficients
        if (self._coeffs is None) and (self._intercepts is None):
            print("Start generating LIME coefficients")
            self._coeffs, self._intercepts = self.get_lime_coefficients(
                factuals_enc_norm
            )
            print("Finished generating LIME coefficients")

        # generate counterfactuals
        for i in range(factuals_enc_norm.shape[0]):
            factual_enc_norm = factuals_enc_norm.iloc[i].T.values
            coeff = self._coeffs[i]
            intercept = self._intercepts[i]

            # Align action set to coefficients
            self._action_set.set_alignment(coefficients=coeff)

            # Build AR flipset
            fs = rs.Flipset(
                x=factual_enc_norm,
                action_set=self._action_set,
                coefficients=coeff,
                intercept=intercept,
            )
            fs_pop = fs.populate(total_items=self._fs_size)

            # Get actions to flip predictions
            actions = fs_pop.actions
            last_action = len(actions) - 1

            for idx, action in enumerate(actions):
                candidate_cf = (factual_enc_norm + action).reshape(
                    (1, -1)
                )  # Reshape to keep two-dim. input
                # Check if candidate counterfactual really flipps the prediction of ML model
                pred_cf = np.argmax(self._mlmodel.predict_proba(candidate_cf))
                pred_f = np.argmax(
                    self._mlmodel.predict_proba(factual_enc_norm.reshape((1, -1)))
                )
                if pred_cf != pred_f:
                    cfs.append(candidate_cf)
                    break

                # If no counterfactual is found apply array with nan values
                if idx == last_action:
                    empty = np.empty(candidate_cf.shape)
                    empty[:] = np.nan
                    cfs.append(empty)

        # Convert output into correct format
        cfs = np.array(cfs).squeeze()
        cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        cfs[self._mlmodel.data.target] = np.argmax(
            self._mlmodel.predict_proba(cfs), axis=1
        )

        return cfs
