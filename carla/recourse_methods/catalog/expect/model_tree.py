from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal

from carla.models.api import MLModel
from carla.models.catalog.trees import get_distilled_model, get_rules
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)

from .hypercube import get_classes_from_rules, get_hypercubes

INF = torch.tensor(float("inf"))


def cdf_difference(upper, lower, value, var, verbose=False):
    normal = Normal(loc=value, scale=torch.sqrt(var))
    upper_cdf = normal.cdf(upper)
    lower_cdf = normal.cdf(lower)
    cdf_diff = upper_cdf - lower_cdf

    if verbose:
        print(f"upper cdf {upper_cdf}")
        print(f"lower cdf {lower_cdf}")
        print(f"cdf diff {cdf_diff}")
        print("--------------------------------")

    if not cdf_diff >= 0:
        raise ValueError(
            "Something went wrong -> Hints: Confused upper with lower?; Are the Hypercubes correct?"
        )

    return cdf_diff


class EXPECTTree(RecourseMethod):
    """
    Implementation of EXPECT from Pawelczyk et. al. [1]_.

    Parameters
    ----------
    mlmodel: carla.model.MLModel
        Black-Box-Model
    hyperparams: dict
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

        * "max_iter": int,
            Number of optimization steps.
        * "optimizer": string
            What optimizer to use.
        * "learning_rate": float
            Learning rate for the optimizer.
        * "var": float
            Variance for the invalidation rate samples.
        * "target_names": list
            List of class names.
        * "lambda": float
            Weight for loss term corresponding to distance for counterfactual.
        * "upper_bound":
            Upper-bound for the data.
        * "lower_bound":
            Lower-bound for the data.
        * "invalidation_target": float
            Target for the invalidation rate.

    .. [1] Pawelczyk, M., Datta, T., van-den-Heuvel, J., Kasneci, G., & Lakkaraju, H. (2022).
            Algorithmic Recourse in the Face of Noisy Human Responses. arXiv preprint arXiv:2203.06768.

    """

    _DEFAULT_HYPERPARAMS = {
        "max_iter": 50,
        "optimizer": "adam",
        "learning_rate": 0.05,
        "var": 0.05,
        "target_names": [1, 0],
        "lambda": 0.2,
        "upper_bound": 1.0,
        "lower_bound": 0.0,
        "invalidation_target": 0.3,
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None):
        super().__init__(mlmodel)
        self.hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.lr = self.hyperparams["learning_rate"]
        self.lam = self.hyperparams["lambda"]
        self.invalidation_target = self.hyperparams["invalidation_target"]
        self.upper_bound = self.hyperparams["upper_bound"]
        self.lower_bound = self.hyperparams["lower_bound"]
        self.feature_names = mlmodel.feature_input_order

        # Get Hypercubes
        model = get_distilled_model(mlmodel)
        rules = get_rules(model, self.feature_names, self.hyperparams["target_names"])
        self.classes = get_classes_from_rules(rules)
        self.all_intervals = get_hypercubes(
            self.feature_names, rules, self.lower_bound, self.upper_bound
        )

    def invalidation_loss(self, x, verbose=False):
        """Compute the invalidation loss.

        Parameters
        ----------
        x:
            Input for which we want to find invalidation loss.
        verbose:
            Print or not print.

        Returns
        -------

        """
        validation_rate = 0
        for i, interval in enumerate(self.all_intervals):
            predicted_label = torch.tensor(self.classes[i]).float()

            if verbose:
                print(f"it: {i}")
                print(f"predicted label: {predicted_label}")

            # only compute loss for positive class labels
            product_cdf_diff = 1
            for j, feature in enumerate(self.feature_names):
                if verbose:
                    print(f"it_f: {j}")

                interval_range = torch.tensor(interval[j][0][1])

                # replace upper and lower bounds due to data constrains
                # TODO should this depend on data normalization (e.g. min max vs scaled)
                if INF in interval_range:
                    interval_range = [
                        y if y != np.inf else self.upper_bound for y in interval_range
                    ]
                if -INF in interval_range:
                    interval_range = [
                        y if y != -np.inf else self.lower_bound for y in interval_range
                    ]

                # form updated intervals
                input_value = x[j].float()
                interval_lower, interval_upper = interval_range[0], interval_range[1]

                d_cdf = cdf_difference(
                    interval_upper,
                    interval_lower,
                    input_value,
                    var=torch.tensor(self.hyperparams["var"]),
                )
                product_cdf_diff *= d_cdf

            validation_rate += predicted_label * product_cdf_diff

        invalidation_rate = torch.tensor(1).float() - validation_rate

        if verbose:
            print(f"validation rate: {validation_rate}")
            print(f"invalidation rate: {invalidation_rate}")

        return invalidation_rate

    def optimization(self, x: torch.tensor):
        x_check = torch.tensor(x, requires_grad=True, dtype=torch.float)

        if self.hyperparams["optimizer"] == "adam":
            optim = torch.optim.Adam([x_check], self.lr)
        else:
            optim = torch.optim.RMSprop([x_check], self.lr)

        for i in range(self.hyperparams["max_iter"]):
            x_check.requires_grad = True

            invalidation_rate = self.invalidation_loss(x_check)

            # Compute & update losses
            loss_invalidation = invalidation_rate - self.invalidation_target

            # Hinge loss
            loss_invalidation[loss_invalidation < 0] = 0

            # Distance loss
            loss_distance = torch.norm(x_check - x, 1)

            # Combined loss
            loss = loss_invalidation + self.lam * loss_distance

            # Update loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            # print(f"invalidation rate iteration {i}: {invalidation_rate.item()}")
            # print(f"loss at iteration {i}: {loss.item()}")
            # print(f"CE suggestion at iteration {i}: {x_check}")

        return x_check.detach().numpy()

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)

        df_cfs = factuals.apply(
            lambda x: self.optimization(torch.from_numpy(x)),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
