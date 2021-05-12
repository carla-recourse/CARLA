from typing import List

import numpy as np


def get_distances(factual: np.ndarray, counterfactual: np.ndarray) -> List[float]:
    """
    Computes distances 1 to 4.
    All features have to be in the same order (without target label).

    Parameters
    ----------
    factual: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    counterfactual: np.ndarray
        Normalized and encoded array with counterfactual data
        Shape: NxM

    Returns
    -------
    list: distances 1 to 4
    """
    if factual.shape != counterfactual.shape:
        raise ValueError("Shapes of factual and counterfactual have to be the same")
    if len(factual.shape) != 2:
        raise ValueError(
            "Shapes of factual and counterfactual have to be 2-dimensional"
        )

    # get difference between original and counterfactual
    delta = get_delta(factual, counterfactual)

    d1 = d1_distance(delta)
    d2 = d2_distance(delta)
    d3 = d3_distance(delta)
    d4 = d4_distance(delta)

    return [d1, d2, d3, d4]


def d1_distance(delta: np.ndarray) -> float:
    """
    Computes D1 distance

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    float
    """
    # compute elements which are greater than 0
    return float(np.sum(delta != 0))


def d2_distance(delta: np.ndarray) -> float:
    """
    Computes D2 distance

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    float
    """

    return float(np.sum(np.abs(delta)))


def d3_distance(delta: np.ndarray) -> float:
    """
    Computes D3 distance

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    float
    """
    return float(np.sum(np.square(np.abs(delta))))


def d4_distance(delta: np.ndarray) -> float:
    """
    Computes D4 distance

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    float
    """
    return np.max(np.abs(delta))


def get_delta(instance: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """
    Compute difference between original instance and counterfactual

    Parameters
    ----------
    instance: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    cf: : np.ndarray
        Normalized and encoded array with counterfactual data.
        Shape: NxM

    Returns
    -------
    np.ndarray
    """
    return cf - instance
