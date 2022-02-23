from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from carla.data.catalog import DataCatalog


def _get_noise_string(node):
    if not node[0] == "x":
        raise ValueError
    return "u" + node[1:]


def _create_synthetic_data(scm, num_samples):
    """
    Generate synthetic data.

    Parameters
    ----------
    scm : CausalModel
        Structural causal model
    num_samples : int
        Number of samples in the dataset

    """

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
            exogenous_variables[_get_noise_string(node)],
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

    df_endogenous = pd.concat([labels, endogenous_variables], axis=1).astype("float64")
    df_exogenous = pd.concat([exogenous_variables], axis=1).astype("float64")
    return df_endogenous, df_exogenous


class ScmDataset(DataCatalog):
    """
    Generate a dataset from structural equations

    Parameters
    ----------
    scm : CausalModel
        Structural causal model
    size : int
        Number of samples in the dataset
    """

    def __init__(
        self,
        scm,
        size: int,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot_drop_binary",
    ):
        # TODO setup normalization with generate_dataset in CausalModel class
        self.scm = scm
        raw, self._noise = _create_synthetic_data(scm, num_samples=size)
        train_raw, test_raw = train_test_split(raw)

        super().__init__(
            scm.scm_class, raw, train_raw, test_raw, scaling_method, encoding_method
        )

    @property
    def categorical(self) -> List[str]:
        """
        Provides the column names of the categorical data.

        Returns
        -------
        List[str]
        """
        return self.scm._categorical

    @property
    def continuous(self) -> List[str]:
        """
        Provides the column names of the continuous data.

        Returns
        -------
        List[str]
        """
        return self.scm._continuous

    @property
    def categorical_noise(self) -> List[str]:
        """
        Provides the column names of the categorical data.

        Returns
        -------
        List[str]
        """
        return self.scm._categorical_noise

    @property
    def continuous_noise(self) -> List[str]:
        """
        Provides the column names of the continuous data.

        Returns
        -------
        List[str]
        """
        return self.scm._continuous_noise

    @property
    def immutables(self) -> List[str]:
        return self.scm._immutables

    @property
    def target(self) -> str:
        return "label"

    @property
    def noise(self) -> pd.DataFrame:
        return self._noise.copy()
