import matplotlib.pyplot as plt

# from .barplot import barplot
from .stripplot import stripplot
from .swarmplot import swarmplot


def _most_important_cont_features(diff, topn):
    """
    Get the top-n most important continuous features

    Parameters
    ----------
    diff: pd.DataFrame
        Difference between counterfactuals and factuals.
    topn: int
        Number of most important features to use.

    Returns
    -------
    Most important features.
    """

    mean = diff.mean()
    variance = diff.var()
    importance = mean + variance
    return importance.sort_values(ascending=False).head(topn).index


def _most_important_cat_features(diff, topn):
    """
    Get the top-n most important categorical features

    Parameters
    ----------
    diff: pd.DataFrame
        Difference between counterfactuals and factuals.
    topn: int
        Number of most important features to use.

    Returns
    -------
    Most important features.
    """
    importance = diff.abs().sum()
    return importance.sort_values(ascending=False).head(topn).index


def plot(factuals, counterfactuals, data, topn):
    """

    Parameters
    ----------
    factuals: pd.DataFrame
        DataFrame containing the factuals.
    counterfactuals: pd.DataFrame
        DataFrame containing the counterfactuals corresponding to above factuals
    data: carla.data.api.Data
        Dataset object. Used for column information.
    topn: int
        Number of most important features to use.

    Returns
    -------

    """

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))

    # Get the continuous and categorical column names
    cont_cols = data.continuous
    cat_cols = list(
        set(counterfactuals.columns) - set(data.continuous) - set(data.target)
    )

    # Compute difference
    diff = counterfactuals - factuals
    cont_cols = _most_important_cont_features(diff[cont_cols], topn)
    cat_cols = _most_important_cat_features(diff[cat_cols], topn)

    swarmplot(diff[cont_cols], factuals[cont_cols], axs[0])
    stripplot(diff[cat_cols], factuals[cat_cols], axs[1])

    # Resize to create overlap
    fig.set_size_inches(10, 10)
