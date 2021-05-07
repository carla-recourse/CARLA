import os

from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, Sequential


class Autoencoder:
    def __init__(
        self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_latent, data_name
    ):
        """
        Defines the structure of the autoencoder
        :param dim_input: int > 0; number of neurons for this layer (for Adult: 104)
        :param dim_hidden_layer_1: int > 0, number of neurons for this layer (for Adult: 30)
        :param dim_hidden_layer_2: int > 0, number of neurons for this layer (for Adult: 15)
        :param dim_latent: int >0; number of dimensions for the latent code
        """

        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.dim_hidden_layer1 = dim_hidden_layer1
        self.dim_hidden_layer2 = dim_hidden_layer2
        self.data_name = data_name

    def train(self, xtrain, xtest, epochs, batch_size):
        def loss(y_true, y_pred):
            """ Negative log likelihood (Bernoulli). """

            # Works if data is normalized between 0 and 1!
            # Keras.losses.binary_crossentropy gives the mean
            # Over the last axis. we require the sum

            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

        x = Input(shape=(self.dim_input,))

        # Encoder
        encoded = Dense(self.dim_hidden_layer1, activation="relu")(x)
        encoded = Dense(self.dim_hidden_layer2, activation="relu")(encoded)
        z = Dense(self.dim_latent, activation="relu")(encoded)

        # Decoder
        decoder = Sequential(
            [
                Dense(
                    self.dim_hidden_layer1, input_dim=self.dim_latent, activation="relu"
                ),
                Dense(self.dim_hidden_layer2, activation="relu"),
                Dense(self.dim_input, activation="sigmoid"),
            ]
        )

        # Compile Autoencoder, Encoder & Decoder
        # encoder = Model(x, z)
        xhat = decoder(z)
        autoencoder = Model(x, xhat)
        autoencoder.compile(optimizer="rmsprop", loss=loss)

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

    def save(self, fitted_ae):

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
