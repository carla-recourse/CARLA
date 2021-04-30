def scaler(fitted_scaler, features, df):
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


def encoder(fitted_encoder, features, df):
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


def order_data(feature_order, df):
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
