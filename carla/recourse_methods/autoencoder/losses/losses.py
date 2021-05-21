import numpy as np
from keras import backend as K


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
