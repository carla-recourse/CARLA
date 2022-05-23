import matplotlib.pyplot as plt

from .barplot import barplot
from .stripplot import stripplot
from .swarmplot import swarmplot


def _most_important_features(diff, topn):
    """
    Get the top-n most important features

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


def single_sample_plot(factual, counterfactual, data, figsize=(7, 7)):
    """
    Create a bar plot for a single sample.

    Parameters
    ----------
    factuals: pd.DataFrame
        DataFrame containing the factuals.
    counterfactuals: pd.DataFrame
        DataFrame containing the counterfactuals corresponding to above factuals
    data: carla.data.api.Data
        Dataset object. Used for column information.
    figsize: tuple(int)
        Figure size.

    Returns
    -------

    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get the continuous and categorical column names
    cont_cols = data.continuous
    cat_cols = [
        c for c in counterfactual.index if c not in data.continuous and c != data.target
    ]

    # Compute difference
    diff = counterfactual - factual
    barplot(diff[cont_cols + cat_cols], ax)


def summary_plot(factuals, counterfactuals, data, topn=5, figsize=(15, 7)):
    """
    Create a swarmplot for the continuous features, and a stripplot for the categorical features.

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
    figsize: tuple(int)
        Figure size.

    Returns
    -------

    """
    # Make the figure bigger first, in order to create overlap later
    fig, axs = plt.subplots(1, 2, figsize=(2 * figsize[0], 2 * figsize[1]))

    # Get the continuous and categorical column names
    cont_cols = data.continuous
    cat_cols = [
        c
        for c in counterfactuals.columns
        if c not in data.continuous and c != data.target
    ]

    # Compute difference
    diff = counterfactuals - factuals
    cont_cols = _most_important_features(diff[cont_cols], topn)
    cat_cols = _most_important_features(diff[cat_cols], topn)

    swarmplot(diff[cont_cols], factuals[cont_cols], axs[0])
    stripplot(diff[cat_cols], factuals[cat_cols], axs[1])

    # Resize to create overlap
    fig.set_size_inches(figsize[0], figsize[1])

    plt.subplots_adjust(wspace=1.0)
