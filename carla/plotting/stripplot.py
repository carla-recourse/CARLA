import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import seaborn as sns


def stripplot(diff, factuals, ax):
    """
    Creates a stripplot. Useful for visualizing the importance of different categorical features for multiple instances.

    Parameters
    ----------
    diff: pd.DataFrame
        Difference between counterfactuals and factuals.
    factuals: pd.DataFrame
        Original factual values.

    Returns
    -------

    """
    jitter = 0.3
    delta = np.random.uniform(-jitter / 2, jitter / 2, len(diff.melt()["value"]))

    """
    Create the figure
    """
    sns.stripplot(
        x=diff.melt()["value"] + delta,
        y=diff.melt()["variable"],
        hue=factuals.melt()["value"],
        palette="seismic",
        jitter=jitter,
        alpha=0.5,
        s=7,
        ax=ax,
    )

    ax.set_xlabel("change")
    ax.set_ylabel("")

    """
    Create the colorbar
    """
    # Get a mappable object with the same colormap as the data
    points = plt.scatter([], [], c=[], vmin=0.0, vmax=1.0, cmap="coolwarm")

    # Make space for the colorbar
    ax.figure.subplots_adjust(right=0.92)

    # Define a new Axes where the colorbar will go
    cax = ax.figure.add_axes([0.94, 0.25, 0.02, 0.6])

    # Remove legend and add colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(points, cax=cax, label="feature value")

    # Change axis
    loc = plticker.MultipleLocator(
        base=1.0
    )  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
