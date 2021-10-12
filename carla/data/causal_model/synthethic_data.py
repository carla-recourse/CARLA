from typing import List

import numpy as np
import pandas as pd

from carla.data.api import Data


def get_noise_string(node):
    if not node[0] == "x":
        raise ValueError
    return "u" + node[1:]


def create_synthetic_data(scm, num_samples):

    exogenous_variables = np.concatenate(
        [
            np.array(scm.noise_distributions[node].sample(num_samples)).reshape(-1, 1)
            for node in scm.get_topological_ordering("exogenous")
        ],
        axis=1,
    )
    exogenous_variables = pd.DataFrame(
        exogenous_variables, columns=scm.get_topological_ordering("exogenous")
    )

    endogenous_variables = exogenous_variables.copy()
    endogenous_variables = endogenous_variables.rename(
        columns=dict(
            zip(
                scm.get_topological_ordering("exogenous"),
                scm.get_topological_ordering("endogenous"),
            )
        )
    )
    # used later to make sure parents are populated when computing children
    endogenous_variables.loc[:] = np.nan
    for node in scm.get_topological_ordering("endogenous"):
        parents = scm.get_parents(node)
        if endogenous_variables.loc[:, list(parents)].isnull().values.any():
            raise ValueError(
                "parents in endogenous_variables should already be occupied"
            )
        endogenous_variables[node] = scm.structural_equations_np[node](
            exogenous_variables[get_noise_string(node)],
            *[endogenous_variables[p] for p in parents],
        )

    # fix a hyperplane
    w = np.ones((endogenous_variables.shape[1], 1))
    # get the average scale of (w^T)*X, this depends on the scale of the data
    scale = 2.5 / np.mean(np.abs(np.dot(endogenous_variables, w)))
    predictions = 1 / (1 + np.exp(-scale * np.dot(endogenous_variables, w)))

    if not 0.20 < np.std(predictions) < 0.42:
        raise ValueError(f"std of labels is strange: {np.std(predictions)}")

    # sample labels from class probabilities in predictions
    uniform_rv = np.random.rand(endogenous_variables.shape[0], 1)
    labels = uniform_rv < predictions
    labels = pd.DataFrame(data=labels, columns={"label"})

    df_non_hot = pd.concat([labels, endogenous_variables, exogenous_variables], axis=1)
    return df_non_hot.astype("float64")


class ScmDataset(Data):
    """
    Generate a dataset from structural equations

    Parameters
    ----------
    scm :
        structural causal model

    Returns
    -------
    None
    """

    def __init__(self, scm, size: int):
        self.name = scm.scm_class
        self._raw = create_synthetic_data(scm, num_samples=size)

    @property
    def categoricals(self) -> List[str]:
        pass

    @property
    def continous(self) -> List[str]:
        pass

    @property
    def immutables(self) -> List[str]:
        pass

    @property
    def target(self) -> str:
        return "label"

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()
