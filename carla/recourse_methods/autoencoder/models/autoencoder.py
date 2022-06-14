import os
from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model, Sequential, model_from_json

from carla.recourse_methods.autoencoder.losses import mse
from carla.recourse_methods.autoencoder.save_load import get_home

tf.compat.v1.disable_eager_execution()


class Autoencoder:
    def __init__(
        self,
        data_name: str,
        layers: Optional[List] = None,
        optimizer: str = "rmsprop",
        loss: Optional[Callable] = None,
    ) -> None:
        """
        Defines the structure of the autoencoder with keras backend

        Parameters
        ----------
        data_name : Name of the dataset. Is used for saving model.
        layers : Depending on the position and number elements, it determines the number and width of layers in the
            form of:
            [input_layer, hidden_layer_1, ...., hidden_layer_n, latent_dimension]
            The encoder structure would be: input_layer -> [hidden_layers] -> latent_dimension
            The decoder structure would be: latent_dimension -> [hidden_layers] -> input_dimension
        loss: Loss function for autoencoder model. Default is Binary Cross Entropy.
        optimizer: Optimizer which is used to train autoencoder model. See keras optimizer.
        """
        if layers is None or self.layers_valid(layers):
            # None layers are used for loading pre-trained models
            self._layers = layers
        else:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        if loss is None:
            self._loss = mse
        else:
            self._loss = loss

        self.data_name = data_name
        self._optimizer = optimizer

    def layers_valid(self, layers: List) -> bool:
        """
        Checks if the layers parameter has at least minimal requirements
        """
        if len(layers) < 2:
            return False

        for layer in layers:
            if layer <= 0:
                return False

        return True

    def train(
        self, xtrain: np.ndarray, xtest: np.ndarray, epochs: int, batch_size: int
    ) -> Model:
        assert (
            self._layers is not None
        )  # Used for mypy to make sure layers are indexable
        x = Input(shape=(self._layers[0],))

        # Encoder
        encoded = Dense(self._layers[1], activation="relu")(x)
        for i in range(2, len(self._layers) - 1):
            encoded = Dense(self._layers[i], activation="relu")(encoded)
        latent_space = Dense(self._layers[-1], activation="relu")(encoded)

        # Decoder
        list_decoder = [
            Dense(self._layers[1], input_dim=self._layers[-1], activation="relu")
        ]
        for i in range(2, len(self._layers) - 1):
            list_decoder.append(Dense(self._layers[i], activation="relu"))
        list_decoder.append(Dense(self._layers[0], activation="sigmoid"))
        decoder = Sequential(list_decoder)

        # Compile Autoencoder, Encoder & Decoder
        # encoder = Model(x, latent_space)
        x_reconstructed = decoder(latent_space)
        autoencoder = Model(x, x_reconstructed)
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

        cache_path = get_home()

        fitted_ae.save_weights(
            os.path.join(
                cache_path,
                "{}_{}.{}".format(self.data_name, fitted_ae.input_shape[1], "h5"),
            )
        )

        # save model
        model_json = fitted_ae.to_json()

        path = os.path.join(
            cache_path,
            "{}_{}.{}".format(self.data_name, fitted_ae.input_shape[1], "json"),
        )

        with open(
            path,
            "w",
        ) as json_file:
            json_file.write(model_json)

    def load(self, input_shape: int) -> Model:
        """
        Loads a pretrained ae from cache

        Parameters
        ----------
        input_shape: determines which model is used

        Returns
        -------

        """
        cache_path = get_home()
        path = os.path.join(
            cache_path,
            "{}_{}".format(self.data_name, input_shape),
        )

        # load ae
        json_file = open(
            "{}.{}".format(path, "json"),
            "r",
        )
        model_ae = model_from_json(json_file.read(), custom_objects={"tf": tf})
        json_file.close()

        model_ae.load_weights("{}.{}".format(path, "h5"))

        # Build layers property from loaded model
        layers = []
        for layer in model_ae.layers[:-1]:
            layers.append(layer.output_shape[1])
        self._layers = layers

        return model_ae
