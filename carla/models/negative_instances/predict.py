from typing import Any

import numpy as np
import pandas as pd

from carla.data.api import Data
from carla.models.pipelining import encode, scale


def predict_negative_instances(model: Any, data: Data) -> pd.DataFrame:
    """Predicts the data target and retrieves the negative instances. (H^-)

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    data : Data
        Dataset used for predictions
    Returns
    -------
    df :  data.api Data() class with negative predicted instances
    """
    df = data.raw
    df["y"] = predict_label(model, data)
    df = df[df["y"] == 0]
    df = df.drop("y", axis="columns")

    return df


def predict_label(model: Any, data: Data, as_prob: bool = False) -> np.ndarray:
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    data : Data
        Dataset used for predictions
    Returns
    -------
    predictions :  2d numpy array with predictions
    """

    # normalize and encode data
    norm_enc_data = scale(model.scaler, data.continous, data.raw)
    norm_enc_data = encode(model.encoder, data.categoricals, norm_enc_data)

    # Keep correct feature order for prediction
    norm_enc_data = norm_enc_data[model.feature_input_order]
    predictions = model.predict(norm_enc_data)

    if not as_prob:
        predictions = predictions.round()

    return predictions
