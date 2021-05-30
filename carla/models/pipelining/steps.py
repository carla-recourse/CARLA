from typing import List

import pandas as pd
from sklearn.base import BaseEstimator


def scale(
    fitted_scaler: BaseEstimator, features: List[str], df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline function to normalize data with fitted sklearn scaler.

    Parameters
    ----------
    fitted_scaler : sklearn Scaler
        Normalizes input data
    features : list
        List of continuous feature
    df : pd.DataFrame
        Data we want to normalize

    Returns
    -------
    output : pd.DataFrame
        Whole DataFrame with normalized values

    """
    output = df.copy()
    output[features] = fitted_scaler.transform(output[features])

    return output


def encode(
    fitted_encoder: BaseEstimator, features: List[str], df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline function to encode data with fitted sklearn OneHotEncoder.

    Parameters
    ----------
    fitted_encoder : sklearn OneHotEncoder
        Encodes input data.
    features : list
        List of categorical feature.
    df : pd.DataFrame
        Data we want to normalize

    Returns
    -------
    output : pd.DataFrame
        Whole DataFrame with encoded values
    """
    output = df.copy()
    encoded_features = fitted_encoder.get_feature_names(features)
    output[encoded_features] = fitted_encoder.transform(output[features])
    output = output.drop(features, axis=1)

    return output


def decode(
    fitted_encoder: BaseEstimator, features: List[str], df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline function to decode data with fitted sklearn OneHotEncoder.

    Parameters
    ----------
    fitted_encoder : sklearn OneHotEncoder
        Encodes input data.
    features : list
        List of categorical feature.
    df : pd.DataFrame
        Data we want to normalize

    Returns
    -------
    output : pd.DataFrame
        Whole DataFrame with encoded values
    """
    output = df.copy()
    encoded_features = fitted_encoder.get_feature_names(features)

    # Prevent errors for datasets without categorical data
    # inverse_transform cannot handle these cases
    if len(encoded_features) == 0:
        return output

    output[features] = fitted_encoder.inverse_transform(output[encoded_features])
    output = output.drop(encoded_features, axis=1)

    return output


def order_data(feature_order: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Restores the correct input feature order for the ML model

    Parameters
    ----------
    feature_order : list
        List of input feature in correct order
    df : pd.DataFrame
        Data we want to order

    Returns
    -------
    output : pd.DataFrame
        Whole DataFrame with ordered feature
    """
    return df[feature_order]
