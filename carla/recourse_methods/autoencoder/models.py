import os
from typing import Callable, List, Optional

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, Sequential


def layers_valid(layers: List) -> bool:
    """
    Checks if the layers parameter has at least minimal requirements

    Returns
    -------
    bool
    """
    if len(layers) < 2:
        return False

    for layer in layers:
        if layer <= 0:
            return False

    return True


class Autoencoder:
    def __init__(
        self,
        layers: List,
        data_name: str,
        optimizer: str = "rmsprop",
        loss: Optional[Callable] = None,
    ) -> None:
        """
        Defines the structure of the autoencoder

        Parameters
        ----------
        layers : list(int > 0)
            Depending on the position and number elements, it determines the number and width of layers in the form of
            [input_layer, hidden_layer_1, ...., hidden_layer_n, latent_dimension]
        data_name : str
            Name of the dataset. Is used for saving model.
        loss: Callable, optional
            Loss function for autoencoder model. Default is Binary Cross Entropy.
        optimizer: str, optional
            Optimizer which is used to train autoencoder model. See keras optimizer.
        """
        if layers_valid(layers):
            self._layers = layers
        else:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        if loss is None:
            self._loss = lambda y_true, y_pred: K.sum(
                K.binary_crossentropy(y_true, y_pred), axis=-1
            )
        else:
            self._loss = loss

        self.data_name = data_name
        self._optimizer = optimizer

    def train(
        self, xtrain: np.ndarray, xtest: np.ndarray, epochs: int, batch_size: int
    ) -> Model:

        x = Input(shape=(self._layers[0],))

        # Encoder
        encoded = Dense(self._layers[1], activation="relu")(x)
        for i in range(2, len(self._layers) - 1):
            encoded = Dense(self._layers[i], activation="relu")(encoded)
        z = Dense(self._layers[-1], activation="relu")(encoded)

        # Decoder
        list_decoder = [
            Dense(self._layers[1], input_dim=self._layers[-1], activation="relu")
        ]
        for i in range(2, len(self._layers) - 1):
            list_decoder.append(Dense(self._layers[i], activation="relu"))
        list_decoder.append(Dense(self._layers[0], activation="sigmoid"))
        decoder = Sequential(list_decoder)

        # Compile Autoencoder, Encoder & Decoder
        # encoder = Model(x, z)
        xhat = decoder(z)
        autoencoder = Model(x, xhat)
        autoencoder.compile(optimizer=self._optimizer, loss=self._loss)

        # Train model
        autoencoder.fit(
            xtrain,
            xtrain,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(xtest, xtest),
            verbose=1,
        )

        return autoencoder

    def save(self, fitted_ae: Model) -> None:

        cache_path = self.get_aes_home()

        fitted_ae.save_weights(
            os.path.join(
                cache_path,
                "{}_{}.{}".format(self.data_name, fitted_ae.input_shape[1], "h5"),
            )
        )

        # save model
        model_json = fitted_ae.to_json()

        with open(
            os.path.join(
                cache_path,
                "{}_{}.{}".format(self.data_name, fitted_ae.input_shape[1], "json"),
            ),
            "w",
        ) as json_file:
            json_file.write(model_json)

    def get_aes_home(self, models_home=None):
        """Return a path to the cache directory for trained autoencoders.

        This directory is then used by :func:`save`.

        If the ``models_home`` argument is not specified, it tries to read from the
        ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

        """

        if models_home is None:
            models_home = os.environ.get(
                "CF_MODELS", os.path.join("~", "carla", "models", "autoencoders")
            )

        models_home = os.path.expanduser(models_home)
        if not os.path.exists(models_home):
            os.makedirs(models_home)

        return models_home
