import os
import re
from urllib.request import urlopen


def get_dataset_names():
    """Report available example datasets, useful for reporting issues.

    Requires an internet connection.

    """

    url = "https://github.com/indyfree/cf-data"
    with urlopen(url) as resp:
        html = resp.read()

    pat = r"/indyfree/cf-data/blob/master/(\w*).csv"
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
