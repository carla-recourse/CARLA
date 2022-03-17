import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod


def yNN(
    counterfactuals: pd.DataFrame,
    recourse_method: RecourseMethod,
    mlmodel: MLModel,
    y: int,
) -> float:
    """

    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of

    Returns
    -------
    float
    """
    number_of_diff_labels = 0
    N = counterfactuals.shape[0]

    nbrs = NearestNeighbors(n_neighbors=y).fit(mlmodel.data.df.values)

    for i, row in counterfactuals.iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = row[mlmodel.data.target]

        for idx in knn:
            neighbour = mlmodel.data.df.iloc[idx]
            neighbour = neighbour.drop(mlmodel.data.target)
            neighbour = neighbour.values.reshape((1, -1))
            neighbour_label = np.argmax(mlmodel.predict_proba(neighbour))

            number_of_diff_labels += np.abs(cf_label - neighbour_label)

    return 1 - (1 / (N * y)) * number_of_diff_labels
