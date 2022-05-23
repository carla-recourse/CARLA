import seaborn as sns


def swarmplot(diff, factuals, ax):
    """
    Creates a swarmplot. Useful for visualizing the importance of different continuous features for multiple instances.

    Parameters
    ----------
    diff: pd.DataFrame
        Difference between counterfactuals and factuals.
    factuals: pd.DataFrame
        Original factual values.

    Returns
    -------

    """

    """
    Create the figure
    """
    sns.swarmplot(
        x="value",
        y="variable",
        data=diff.melt(),
        hue=factuals.melt()["value"],
        alpha=0.5,
        size=7,
        palette="coolwarm",
        ax=ax,
    )

    ax.set_xlabel("change")
    ax.set_ylabel("features")

    """
    Create the colorbar
    """
    # # Get a mappable object with the same colormap as the data
    # points = plt.scatter([], [], c=[], vmin=0.0, vmax=1.0, cmap="coolwarm")

    # Make space for the colorbar
    ax.figure.subplots_adjust(right=0.92)
    #
    # # Define a new Axes where the colorbar will go
    # cax = ax.figure.add_axes([0.94, 0.25, 0.02, 0.6])

    # Remove legend and add colorbar
    ax.get_legend().remove()
    # ax.figure.colorbar(points, cax=cax, label="feature value")
