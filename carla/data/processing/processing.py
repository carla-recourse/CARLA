from sklearn import preprocessing


def normalize(df, num_cols, scaler=None):
    """
    Normalizes continous columns of the given dataframe

    :param df: Datafame to scale
    :param num_cols: Numerical columns to scale
    :param scaler: Prefitted Sklearn Scaler
    :return: Data
    """
    df_scale = df.copy()

    if not scaler:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df_scale[num_cols])

    df_scale[num_cols] = scaler.transform(df_scale[num_cols])

    return df_scale
