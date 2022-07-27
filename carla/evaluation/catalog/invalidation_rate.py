import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal

from carla.evaluation.api import Evaluation


class InvalidationRate(Evaluation):
    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.var = hyperparameters["var"]
        self.n_samples = hyperparameters["n_samples"]
        self.cf_label = hyperparameters["cf_label"]
        self.columns = ["invalidation_rate"]

    def invalidation_rate(self, x):
        x = x.float()
        Sigma = torch.eye(len(x)) * self.var

        # Monte Carlo Invalidation Loss
        random_samples = MultivariateNormal(loc=x, covariance_matrix=Sigma).rsample(
            (self.n_samples,)
        )
        random_samples = random_samples.cpu().detach().numpy()
        y_hat = self.mlmodel.predict_proba(random_samples)[:, 1]
        invalidation_rate = self.cf_label - np.mean(y_hat)

        return invalidation_rate

    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        ir_rates = counterfactuals.apply(
            lambda x: self.invalidation_rate(torch.from_numpy(x)), raw=True, axis=1
        )
        return pd.DataFrame(ir_rates, columns=self.columns)
