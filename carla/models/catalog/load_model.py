import os
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlretrieve

import joblib
import tensorflow as tf
import torch

PYTORCH_EXT = "pt"
TENSORFLOW_EXT = "h5"
SKLEARN_EXT = "skjoblib"
XGBOOST_EXT = "xgjoblib"


def load_online_model(
    name: str,
    dataset: str,
    ext: str,
    cache: bool = True,
    models_home: Optional[str] = None,
    **kws,
):
    """Load an pretrained model from the online repository (requires internet).

    This function provides quick access to a number of models trained on example
    datasets that are commonly useful for evaluating counterfactual methods.

    Note that the models have been trained on the example datasets hosted on
    https://github.com/carla-recourse/cf-models.


    Use :func:`get_model_names` to see a list of available models given the dataset.

    Parameters
    ----------
    name: str
        Name of the model ``{name}.{ext}`` on https://github.com/carla-recourse/cf-models.
    dataset: str
        Name of the dataset the model has been trained on.
    ext: str
        Extension of the file.
    cache: boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    models_home: string, optional
        The directory in which to cache data; see :func:`get_models_home`.
    kws: keys and values, optional
        Additional keyword arguments are passed to passed through to the read model function
    Returns
    -------
    model :  Tensorflow or PyTorch model
    """
    full_path = (
        "https://raw.githubusercontent.com/"
        "carla-recourse/cf-models/change-pytorch-models/models/"
        f"{dataset}/{name}.{ext}"
    )

    if cache:
        cache_path = os.path.join(
            get_models_home(models_home), dataset, os.path.basename(full_path)
        )

        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                urlretrieve(full_path, cache_path)
            except HTTPError as e:
                raise ValueError(
                    f"'{name}' is not an available model for dataset '{dataset}'.", e
                )

        full_path = cache_path

    if ext == PYTORCH_EXT:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.load(full_path, map_location=device)
    elif ext == TENSORFLOW_EXT:
        model = tf.keras.models.load_model(full_path, compile=False)
    elif ext == SKLEARN_EXT:
        model = joblib.load(full_path)
    elif ext == XGBOOST_EXT:
        model = joblib.load(full_path)
    else:
        raise NotImplementedError("Extension not supported:", ext)

    return model


def load_trained_model(
    save_name: str,
    data_name: str,
    backend: str,
    models_home: Optional[str] = None,
):
    """
    Try to load a trained model from disk, else return None.

    Parameters
    ----------
    save_name: str
        The filename which is used for the saved model.
    data_name: str
        The subfolder which the model is saved in, corresponding to the dataset.
    backend : {'tensorflow', 'pytorch', 'sklearn', 'xgboost'}
        Specifies the used framework.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.

    Returns
    -------
    None if not able to load model, else returns loaded model.
    """
    # set model extension
    if backend == "pytorch":
        ext = PYTORCH_EXT
    elif backend == "tensorflow":
        ext = TENSORFLOW_EXT
    elif backend == "sklearn":
        ext = SKLEARN_EXT
    elif backend == "xgboost":
        ext = XGBOOST_EXT
    else:
        raise NotImplementedError("Backend not supported:", backend)

    # save location
    cache_path = os.path.join(
        get_models_home(models_home), data_name, f"{save_name}.{ext}"
    )

    if os.path.exists(cache_path):
        # load the model
        if backend == "pytorch":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load(cache_path, map_location=device)
        elif backend == "tensorflow":
            model = tf.keras.models.load_model(cache_path, compile=False)
        elif backend == "sklearn" or backend == "xgboost":
            model = joblib.load(cache_path)
        else:
            raise NotImplementedError("Backend not supported:", backend)
        print(f"Loaded model from {cache_path}")
        return model


def save_model(
    model,
    save_name: str,
    data_name: str,
    backend: str,
    models_home: Optional[str] = None,
):
    """
    Save a model to disk.

    Parameters
    ----------
    model: classifier model
        Model that we want to save to disk.
    save_name: str
        The filename which is used for the saved model.
    data_name: str
        The subfolder which the model is saved in, corresponding to the dataset.
    backend : {'tensorflow', 'pytorch', 'sklearn', 'xgboost'}
        Specifies the used framework.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.

    Returns
    -------
    None
    """
    # set model extension
    if backend == "pytorch":
        ext = PYTORCH_EXT
    elif backend == "tensorflow":
        ext = TENSORFLOW_EXT
    elif backend == "sklearn":
        ext = SKLEARN_EXT
    elif backend == "xgboost":
        ext = XGBOOST_EXT
    else:
        raise NotImplementedError("Backend not supported:", backend)

    # save location
    cache_path = os.path.join(
        get_models_home(models_home), data_name, f"{save_name}.{ext}"
    )
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # save the model
    if backend == "pytorch":
        torch.save(model, cache_path)
    elif backend == "tensorflow":
        model.save(cache_path)
    elif backend == "sklearn" or backend == "xgboost":
        joblib.dump(model, cache_path)


def get_models_home(models_home=None):
    """Return a path to the cache directory for example models.

    This directory is then used by :func:`load_model`.

    If the ``models_home`` argument is not specified, it tries to read from the
    ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

    """

    if models_home is None:
        models_home = os.environ.get("CF_MODELS", os.path.join("~", "carla", "models"))

    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home
