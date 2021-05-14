from typing import List

import numpy as np
import pandas as pd


def get_distances(
    data: pd.DataFrame, factual: List, counterfactual: List
) -> np.ndarray:
    """
    Computes distances 1 to 4
    :param data: Dataframe with original data
    :param factual: List of features
    :param counterfactual: List of features
    :return: Array of distances 1 to 4
    """
    d1 = d1_distance(factual, counterfactual)
    d2 = d2_distance(factual, counterfactual, data)
    d3 = d3_distance(factual, counterfactual, data)
    d4 = d4_distance(factual, counterfactual)

    return np.array([d1, d2, d3, d4])


def d1_distance(instance: List, cf: List) -> float:
    """
    Compute d1-distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)

    # compute elements which are greater than 0
    delta_bin = [i != 0 for i in delta]
    delta_bin = delta_bin[:-1]  # loose label column

    d1 = sum(delta_bin)

    return d1


def d2_distance(instance: List, cf: List, df: pd.DataFrame) -> float:
    """
    Compute d2 distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :param df: Dataframe object of dataset
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)
    delta = delta[:-1]  # loose label column

    # get range of every feature
    range = get_range(df)

    d2_list = [np.abs(x[0] / x[1]) for x in zip(delta, range)]
    d2 = sum(d2_list)

    return d2


def d3_distance(instance: List, cf: List, df: pd.DataFrame) -> float:
    """
    Compute d3 distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :param df: Dataframe object of dataset
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)
    delta = delta[:-1]  # loose label column

    # get range of every feature
    range = get_range(df)

    d3_list = [(x[0] / x[1]) ** 2 for x in zip(delta, range)]
    d3 = sum(d3_list)

    return d3


def d4_distance(instance: List, cf: List) -> float:
    """
    Compute d4 distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)
    delta = delta[:-1]  # loose label column

    d4_list = [np.abs(x) for x in delta]
    d4 = np.max(d4_list)

    return d4


def get_delta(instance: List, cf: List) -> List:
    """
    Compute difference between original instance and counterfactual
    :param instance: List of features of original instance
    :param cf: List of features of counterfactual
    :return: List of differences between cf and original instance
    """
    delta = []
    for i, original in enumerate(instance):
        counterfactual = cf[i]

        if type(original) == str:
            if original == counterfactual:
                delta.append(0)
            else:
                delta.append(1)
        else:
            delta.append(counterfactual - original)

    return delta


def get_max_list(data: np.ndarray) -> List[int]:
    """
    get max element for every column.
    Max for string elements is 1
    :param data: numpy array
    :return: list of max elements
    """
    max = []
    for i in range(data.shape[-1] - 1):
        column = data[:, i]

        if type(column[0]) == str:
            max.append(1)
        else:
            max.append(np.max(column))

    return max


def get_min_list(data: np.ndarray) -> List[int]:
    """
    get min element for every column.
    Min for string elements is 0
    :param data: numpy array
    :return: list of min elements
    """
    min = []
    for i in range(data.shape[-1] - 1):
        column = data[:, i]

        if type(column[0]) == str:
            min.append(0)
        else:
            min.append(np.min(column))

    return min


def get_range(df: pd.DataFrame) -> List[int]:
    """
    Get range max - min of every feature
    :param df: dataframe object of dataset
    :return: list of ranges for every feature
    """
    data = df.values
    max = get_max_list(data)
    min = get_min_list(data)

    range = [x[0] - x[1] for x in zip(max, min)]

    return range
