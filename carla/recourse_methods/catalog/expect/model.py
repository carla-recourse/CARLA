from typing import Dict

import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)


class EXPECT(RecourseMethod):
    """
    Implementation of EXPECT from TODO

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
        * "norm": {1, 2}
            Norm to use to compute distance for counterfactual.
        * "lambda": float
            Weight for loss term correspoding to distance for counterfactual.
        * "n_samples":
            Number of samples for the invalidation rate samples.
        * "invalidation_target": float
            Target for the invalidation rate.

    """

    _DEFAULT_HYPERPARAMS = {
        "max_iter": 150,
        "optimizer": "adam",
        "learning_rate": 0.05,
        "var": 0.05,
        "norm": 1,
        "lambda": 0.2,
        "n_samples": 10000,
        "invalidation_target": 0.3,
        "EXPECT": True,
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None):
        super().__init__(mlmodel)
        self.hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.lr = self.hyperparams["learning_rate"]
        self.EXPECT = self.hyperparams["EXPECT"]
        self.lam = self.hyperparams["lambda"]
        self.invalidation_target = self.hyperparams["invalidation_target"]
        self.loss_fn = torch.nn.BCELoss()

    def invalidation_rate(self, x):
        Sigma = torch.eye(len(x)) * self.hyperparams["var"]

        # Monte Carlo Invalidation Loss
        random_samples = MultivariateNormal(loc=x, covariance_matrix=Sigma).rsample(
            (self.hyperparams["n_samples"],)
        )
        y_hat = self._mlmodel.predict_proba(random_samples)[:, 1]
        invalidation_rate = 1 - torch.mean(y_hat, dim=0)

        return invalidation_rate

    def optimization(self, x: torch.tensor):
        x_check = torch.tensor(x, requires_grad=True, dtype=torch.float)

        if self.hyperparams["optimizer"] == "adam":
            optim = torch.optim.Adam([x_check], self.lr)
        else:
            optim = torch.optim.RMSprop([x_check], self.lr)

        for i in range(self.hyperparams["max_iter"]):
            x_check.requires_grad = True

            # Distance loss
            loss_distance = torch.norm(x_check - x, self.hyperparams["norm"])

            # BCE loss
            pred = self._mlmodel.predict_proba(x_check[None, ...]).squeeze(0)
            target = torch.tensor([0.00, 1.00], device=pred.device)

            # Combined loss
            loss = self.loss_fn(pred, target) + self.lam * loss_distance

            if self.EXPECT:
                invalidation_rate = self.invalidation_rate(x_check)
                # Compute & update losses
                loss_invalidation = invalidation_rate - self.invalidation_target
                # Hinge loss
                loss_invalidation[loss_invalidation < 0] = 0

                print(f"invalidation rate iteration {i}: {invalidation_rate.item()}")

                loss += 2 * loss_invalidation

            # Update loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self._mlmodel.predict_proba(x_check[None, ...])[:, 1] > 0.5:
                # A implies B <=> not(A) or B
                if not self.EXPECT or (invalidation_rate < self.invalidation_target):
                    break

        return x_check.detach().numpy()

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)

        df_cfs = factuals.apply(
            lambda x: self.optimization(torch.from_numpy(x)), raw=True, axis=1
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
