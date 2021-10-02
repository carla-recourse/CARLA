import os
import re
from typing import Any, List
from urllib.request import urlopen, urlretrieve

import pandas as pd


def load_dataset(
    name: str, cache: bool = True, data_home: str = None, **kws
) -> pd.DataFrame:
    """Load an example dataset from the online repository (requires internet).

    This function provides quick access to a number of example datasets
    that are commonly useful for evaluating counterfatual methods.

    Note that some of the datasets have a small amount of preprocessing applied
    to define a proper ordering for categorical variables.

    Use :func:`get_dataset_names` to see a list of available datasets.

    Parameters
    ----------
    name : str
        Name of the dataset ``{name}.csv`` on https://github.com/carla-recourse/cf-data.
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    """

    path = "https://raw.githubusercontent.com/carla-recourse/cf-data/master/{}.csv"
    full_path = path.format(name)

    if cache:
        cache_path = os.path.join(get_data_home(data_home), os.path.basename(full_path))

        if not os.path.exists(cache_path):
            if name not in get_dataset_names():
                raise ValueError(f"'{name}' is not an available dataset.")
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    df = pd.read_csv(full_path, **kws)

    if df.iloc[-1].isnull().all():
        df = df.iloc[:-1]

    # TODO: Only until NANs are not longer in the dataset (issue #28)
    df = df.dropna()

    return df


def get_dataset_names() -> List[Any]:
    """Report available example datasets, useful for reporting issues.

    Requires an internet connection.

    """

    url = "https://github.com/carla-recourse/cf-data"
    with urlopen(url) as resp:
        html = resp.read()

    pat = r"/carla-recourse/cf-data/blob/main/(\w*).csv"
    datasets = re.findall(pat, html.decode())
    return datasets


def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.

    This directory is then used by :func:`load_dataset`.

    If the ``data_home`` argument is not specified, it tries to read from the
    ``CF_DATA`` environment variable and defaults to ``~/cf-data``.

    """

    if data_home is None:
        data_home = os.environ.get("CF_DATA", os.path.join("~", "cf-data"))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home
