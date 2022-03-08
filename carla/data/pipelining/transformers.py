from sklearn import preprocessing


def fit_scaler(scaling_method, df):
    """

    Parameters
    ----------
    scaling_method: {"MinMax", "Standard", "Identity"}
        String indicating what scaling method to use.
    df: pd.DataFrame
        DataFrame only containing continuous data.

    Returns
    -------
    sklearn.base.BaseEstimator

    """
    # TODO rather then passing a string this could accept a function
    if scaling_method == "MinMax":
        fitted_scaler = preprocessing.MinMaxScaler().fit(df)
    elif scaling_method == "Standard":
        fitted_scaler = preprocessing.StandardScaler().fit(df)
    elif scaling_method is None or "Identity":
        fitted_scaler = preprocessing.FunctionTransformer(func=None, inverse_func=None)
    else:
        raise ValueError("Scaling Method not known")
    return fitted_scaler


def fit_encoder(encoding_method, df):
    """

    Parameters
    ----------
    encoding_method: {"OneHot", "OneHot_drop_binary", "Identity"}
        String indicating what encoding method to use.
    df: pd.DataFrame
        DataFrame containing only categorical data.

    Returns
    -------
    sklearn.base.BaseEstimator

    """
    # TODO rather then passing a string this could accept a function
    if encoding_method == "OneHot":
        fitted_encoder = preprocessing.OneHotEncoder(
            handle_unknown="error", sparse=False
        ).fit(df)
    elif encoding_method == "OneHot_drop_binary":
        fitted_encoder = preprocessing.OneHotEncoder(
            drop="if_binary", handle_unknown="error", sparse=False
        ).fit(df)
    elif encoding_method is None or "Identity":
        fitted_encoder = preprocessing.FunctionTransformer(func=None, inverse_func=None)
    else:
        raise ValueError("Encoding Method not known")
    return fitted_encoder
