"""

code adapted from:
https://github.com/a-lucic/focus
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.tree import DecisionTreeClassifier

from carla.models.api import MLModel
from carla.models.catalog import trees
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.focus.distances import distance_func
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)


def _filter_hinge_loss(n_class, mask_vector, features, sigma, temperature, model_fn):
    """
    This functions as the prediction loss

    Parameters
    ----------
    n_class : number of classes
    mask_vector : 0 if prediction is flipped, 1 otherwise
    features : current (perturbed) input features
    sigma: float
    temperature: float
    model_fn: model function

    Returns
    -------
    hinge loss
    """
    n_input = features.shape[0]

    # if mask_vector all 0, i.e. all labels flipped
    if not np.any(mask_vector):
        return np.zeros((n_input, n_class))

    # filters feature input based on the mask
    filtered_input = tf.boolean_mask(features, mask_vector)

    # if sigma or temperature are not scalars
    if type(sigma) != float or type(sigma) != int:
        sigma = tf.boolean_mask(sigma, mask_vector)
    if type(temperature) != float or type(temperature) != int:
        temperature = tf.boolean_mask(temperature, mask_vector)

    # compute loss
    filtered_loss = model_fn(filtered_input, sigma, temperature)

    indices = np.where(mask_vector)[0]
    zero_loss = np.zeros((n_input, n_class))
    # add sparse updates to an existing tensor according to indices
    hinge_loss = tf.tensor_scatter_nd_add(
        tensor=zero_loss, indices=indices[:, None], updates=filtered_loss
    )
    return hinge_loss


class FOCUS(RecourseMethod):
    """
    Implementation of Focus [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    checked_hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "optimizer": {"adam", "gd"}
            Determines the optimizer.
        * "n_class": int
            Number of classes.
        * "n_iter": int
            Number of iterations to run for.
        * "sigma": float
            Parameter in sig(z) = (1 + exp(sigma * z)^-1, controls degree of approximation.
        * "temperature": float
            Parameter in the softmax operation, also controls degreee of approximation.
        * "distance_weight": float
            Determines the weight of the counterfactual distance in the loss.
        * "distance_func": {"l1", "l2"}
            Norm to be used.

    .. [1] Lucic, A., Oosterhuis, H., Haned, H., & de Rijke, M. (2018). FOCUS: Flexible optimizable counterfactual
            explanations for tree ensembles. arXiv preprint arXiv:1910.12199.
    """

    _DEFAULT_HYPERPARAMS = {
        "optimizer": "adam",
        "lr": 0.001,
        "n_class": 2,
        "n_iter": 1000,
        "sigma": 1.0,
        "temperature": 1.0,
        "distance_weight": 0.01,
        "distance_func": "l1",
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:

        supported_backends = ["sklearn", "xgboost"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self.model = mlmodel

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        if checked_hyperparams["optimizer"] == "adam":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=checked_hyperparams["lr"]
            )
        elif checked_hyperparams["optimizer"] == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=checked_hyperparams["lr"]
            )

        self.n_class = checked_hyperparams["n_class"]
        self.n_iter = checked_hyperparams["n_iter"]
        self.sigma_val = checked_hyperparams["sigma"]
        self.temp_val = checked_hyperparams["temperature"]
        self.distance_weight_val = checked_hyperparams["distance_weight"]
        self.distance_function = checked_hyperparams["distance_func"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:

        best_perturb = np.array([])

        def f(best_perturb):
            # doesn't work with categorical features, so they aren't used
            original_input = self.model.get_ordered_features(factuals)
            original_input = original_input.to_numpy()
            ground_truth = self.model.predict(original_input)

            # these will be the perturbed features, i.e. counterfactuals
            perturbed = tf.Variable(
                initial_value=original_input, name="perturbed_features", trainable=True
            )
            to_optimize = [perturbed]

            class_index = np.zeros(len(original_input), dtype=np.int64)
            for i, class_name in enumerate(self.model.raw_model.classes_):
                mask = np.equal(ground_truth, class_name)
                class_index[mask] = i
            class_index = tf.constant(class_index, dtype=tf.int64)
            example_range = tf.constant(np.arange(len(original_input), dtype=np.int64))
            example_class_index = tf.stack((example_range, class_index), axis=1)

            # booleans to indicate if label has flipped
            indicator = np.ones(len(factuals))

            # hyperparameters
            sigma = np.full(len(factuals), self.sigma_val)
            temperature = np.full(len(factuals), self.temp_val)
            distance_weight = np.full(len(factuals), self.distance_weight_val)

            best_distance = np.full(len(factuals), 1000.0)
            best_perturb = np.zeros(perturbed.shape)

            for i in range(self.n_iter):
                with tf.GradientTape(persistent=True) as t:
                    p_model = _filter_hinge_loss(
                        self.n_class,
                        indicator,
                        perturbed,
                        sigma,
                        temperature,
                        self._prob_from_input,
                    )
                    approx_prob = tf.gather_nd(p_model, example_class_index)

                    eps = 10.0**-10
                    distance = distance_func(
                        self.distance_function, perturbed, original_input, eps
                    )

                    # the losses
                    prediction_loss = indicator * approx_prob
                    distance_loss = distance_weight * distance
                    total_loss = tf.reduce_mean(prediction_loss + distance_loss)
                    # optimize the losses
                    grad = t.gradient(total_loss, to_optimize)
                    self.optimizer.apply_gradients(
                        zip(grad, to_optimize),
                        global_step=tf.compat.v1.train.get_or_create_global_step(),
                    )
                    # clip perturbed values between 0 and 1 (inclusive)
                    tf.compat.v1.assign(
                        perturbed, tf.math.minimum(1, tf.math.maximum(0, perturbed))
                    )

                    true_distance = distance_func(
                        self.distance_function, perturbed, original_input, 0
                    ).numpy()

                    # get the class predictions for the perturbed features
                    current_predict = self.model.predict(perturbed.numpy())
                    indicator = np.equal(ground_truth, current_predict).astype(
                        np.float64
                    )

                    # get best perturbation so far, did prediction flip
                    mask_flipped = np.not_equal(ground_truth, current_predict)
                    # is distance lower then previous best distance
                    mask_smaller_dist = np.less(true_distance, best_distance)

                    # update best distances
                    temp_dist = best_distance.copy()
                    temp_dist[mask_flipped] = true_distance[mask_flipped]
                    best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

                    # update best perturbations
                    temp_perturb = best_perturb.copy()
                    temp_perturb[mask_flipped] = perturbed[mask_flipped]
                    best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

            return best_perturb

        # Little bit hacky, but needed as other tf code is graph based.
        # Graph based tf and eager execution for tf don't work together nicely.
        with tf.compat.v1.Session() as sess:
            pf = tfe.py_func(f, [best_perturb], tf.float32)
            best_perturb = sess.run(pf)

        df_cfs = pd.DataFrame(best_perturb, columns=self.model.data.continuous)
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs

    def _prob_from_input(self, perturbed, sigma, temperature):
        feat_columns = self.model.data.continuous
        if not isinstance(self.model.raw_model, DecisionTreeClassifier):
            return trees.get_prob_classification_forest(
                self.model,
                feat_columns,
                perturbed,
                sigma=sigma,
                temperature=temperature,
            )
        elif isinstance(self.model.raw_model, DecisionTreeClassifier):
            return trees.get_prob_classification_tree(
                self.model.raw_model, feat_columns, perturbed, sigma=sigma
            )
