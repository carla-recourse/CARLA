import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from carla.models.api import MLModel


def yNN(
    counterfactuals: pd.DataFrame,
    mlmodel: MLModel,
    y: int,
    cf_label: int,
) -> float:
    """

    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    mlmodel: Classification model
    y: Number of neighbors
    cf_label: The target counterfactual label

    Returns
    -------
    float
    """
    number_of_diff_labels = 0
    N = counterfactuals.shape[0]

    factuals = mlmodel.get_ordered_features(mlmodel.data.df)
    nbrs = NearestNeighbors(n_neighbors=y).fit(factuals.values)

    for i, row in counterfactuals.iterrows():

        if np.any(row.isna()):
            raise ValueError(f"row {i} did not contain a valid counterfactual")

        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        for idx in knn:
            neighbour = factuals.iloc[idx]
            neighbour = neighbour.values.reshape((1, -1))
            neighbour_label = np.argmax(mlmodel.predict_proba(neighbour))

            number_of_diff_labels += np.abs(cf_label - neighbour_label)

    return 1 - (1 / (N * y)) * number_of_diff_labels
