import os
import re
from urllib.request import urlopen, urlretrieve

import tensorflow as tf
import torch


def load_model(name, dataset, engine="torch", cache=True, models_home=None, **kws):
    """Load an pretrained model from the online repository (requires internet).

    This function provides quick access to a number of models trained on example
    datasets that are commonly useful for evaluating counterfatual methods.

    Note that the models have been trained on the example datasets hosted on
    https://github.com/indyfree/cf-models.


    Use :func:`get_model_names` to see a list of available models given the dataset.

    Parameters
    ----------
    name : str
        Name of the model ``{name}.{ext}`` on https://github.com/indyfree/cf-models.
    dataset : str
        Name of the dataset the model has been trained on.
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to the read model function
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    """
    if engine == "torch":
        ext = "pt"
    elif engine == "tensorflow":
        ext = "hd5"
    else:
        raise NotImplementedError(f"'{engine}' is not an supported engine.")

    full_path = (
        "https://raw.githubusercontent.com/"
        "indyfree/cf-models/main/models/"
        f"{dataset}/{name}.{ext}"
    )

    if cache:
        cache_path = os.path.join(
            get_models_home(models_home), os.path.basename(full_path)
        )

        if not os.path.exists(cache_path):
            if name not in get_model_names(dataset):
                raise ValueError(f"'{name}' is not an available model.")
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    if engine == "torch":
        model = torch.load(full_path).eval()
    elif engine == "tensorflow":
        model = tf.keras.models.load_model(full_path)

    return model


def get_model_names(dataset):
    """Report available example models, useful for reporting issues.

    Requires an internet connection.

    """

    url = f"https://github.com/indyfree/cf-models/tree/main/models/{dataset}"
    with urlopen(url) as resp:
        html = resp.read()

    pat = "/indyfree/cf-models/blob/main/" f"models/{dataset}/" r"(\w*).pt"
    datasets = re.findall(pat, html.decode())
    return datasets


def get_models_home(models_home=None):
    """Return a path to the cache directory for example models.

    This directory is then used by :func:`load_model`.

    If the ``models_home`` argument is not specified, it tries to read from the
    ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

    """

    if models_home is None:
        models_home = os.environ.get(
            "CF_MODELS", os.path.join("~", "cf-benchmark", "models")
        )

    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home
