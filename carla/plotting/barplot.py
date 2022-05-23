import seaborn as sns
from matplotlib import cm


def barplot(diff, ax):
    """
    Create a horizontal barplot. Useful for visualizing feature importance for a single instance.

    Parameters
    ----------
    diff: pd.DataFrame
        Difference between counterfactuals and factuals.

    Returns
    -------

    """
    # Color code the features
    cmap = cm.get_cmap("coolwarm")
    colors = [cmap(0.0) if diff[i] < 0 else cmap(1.0) for i in range(len(diff))]

    """
    Create the plot
    """
    sns.barplot(x=diff.values, y=diff.index, palette=colors, ax=ax)
    ax.set_xlabel("change")
    ax.set_ylabel("feature")
