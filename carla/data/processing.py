from sklearn import preprocessing


def normalize(df, num_cols, scaler=None):
    """
    Normalizes continous columns of the given dataframe

    :param df: Datafame to scale
    :param num_cols: Numerical columns to scale
    :param scaler: Prefitted Sklearn Scaler
    :return: Data
    """

    if not scaler:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df[num_cols])

    df[num_cols] = scaler.transform(df[num_cols])

    return df
