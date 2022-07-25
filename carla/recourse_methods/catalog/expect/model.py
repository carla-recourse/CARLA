from typing import Dict

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
import pandas as pd

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from carla.recourse_methods.processing import (
    merge_default_parameters,
    check_counterfactuals,
)


class EXPECT(RecourseMethod):

    _DEFAULT_HYPERPARAMS = {
        "max_iter": 150,
        "optimizer": "adam",
        "learning_rate": 0.05,
        "sigma": 0.05,
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

    def optimization(self, x: torch.tensor):
        sigma = torch.eye(len(x)) * self.hyperparams["sigma"]
        x_check = torch.tensor(x, requires_grad=True, dtype=torch.float)

        if self.hyperparams["optimizer"] == "adam":
            optim = torch.optim.Adam([x_check], self.lr)
        else:
            optim = torch.optim.RMSprop([x_check], self.lr)

        for i in range(self.hyperparams["max_iter"]):
            x_check.requires_grad = True

            if self.EXPECT:
                # Monte Carlo Invalidation Loss
                random_samples = MultivariateNormal(
                    loc=x_check, covariance_matrix=sigma
                ).rsample((self.hyperparams["n_samples"],))
                y_hat = self._mlmodel.predict_proba(random_samples)[:, 1]
                invalidation_rate = 1 - torch.mean(y_hat, dim=0)

                # Compute & update losses
                loss_invalidation = invalidation_rate - self.invalidation_target
                # Hinge loss
                loss_invalidation[loss_invalidation < 0] = 0

                print(f"invalidation rate iteration {i}: {invalidation_rate.item()}")

            # Distance loss
            loss_distance = torch.norm(x_check - x, self.hyperparams["norm"])

            # BCE loss
            pred = self._mlmodel.predict_proba(x_check[None, ...]).squeeze(0)
            target = torch.tensor([0.00, 1.00], device=pred.device)

            # Combined loss
            loss = self.loss_fn(pred, target) + self.lam * loss_distance
            if self.EXPECT:
                loss += 2 * loss_invalidation

            # Update loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self._mlmodel.predict_proba(x_check[None, ...])[:, 1] > 0.5:
                # A implies B <=> not(A) or B
                if not self.EXPECT or (invalidation_rate < self.invalidation_target):
                    break

        return x_check.detach().numpy()  # , invalidation_rate.detach().numpy()

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)

        df_cfs = factuals.apply(
            lambda x: self.optimization(torch.from_numpy(x)), raw=True, axis=1
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
