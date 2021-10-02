import copy
import os
from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras.layers import Dense, Input
from keras.models import Model, Sequential, model_from_json
from torch import optim
from tqdm import trange

from carla import log
from carla.recourse_methods.autoencoder.dataloader import VAEDataset
from carla.recourse_methods.autoencoder.losses import binary_crossentropy, csvae_loss
from carla.recourse_methods.autoencoder.save_load import get_home


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
            self._loss = binary_crossentropy
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


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        data_name: str,
        layers: List,
    ):
        super(VariationalAutoencoder, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self._data_name = data_name
        self._input_dim = layers[0]
        latent_dim = layers[-1]

        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.BatchNorm1d(layers[i]))
            lst_encoder.append(nn.ReLU())
        encoder = nn.Sequential(*lst_encoder)

        self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append(nn.BatchNorm1d(layers[i]))
            lst_decoder.append((nn.ReLU()))
        decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(
            decoder,
            nn.Linear(layers[1], self._input_dim),
            nn.BatchNorm1d(self._input_dim),
            nn.Sigmoid(),
        )

        self.log_var_dec = nn.Sequential(
            decoder,
            nn.Linear(layers[1], self._input_dim),
            nn.BatchNorm1d(self._input_dim),
            nn.Sigmoid(),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def encode(self, x):
        return self._mu_enc(x), self._log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)

    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z_rep = self.__reparametrization_trick(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z_rep)

        return mu_x, log_var_x, z_rep, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)

    def regenerate(self, z):
        mu_x, log_var_x = self.decode(z)
        return mu_x

    def VAE_loss(self, mse_loss, mu, logvar):
        MSE = mse_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    def fit(
        self, xtrain: np.ndarray, lambda_reg=1e-6, epochs=5, lr=1e-3, batch_size=32
    ):
        train_set = VAEDataset(xtrain)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )

        criterion = nn.MSELoss()

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        log.info("Start training of Variational Autoencoder...")
        for epoch in range(epochs):

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0

            # Train for all the batches
            for data, _ in train_loader:
                data = data.view(data.shape[0], -1)

                # forward pass
                MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = self(
                    data
                )

                reconstruction = MU_X_eval
                mse_loss = criterion(reconstruction, data)
                loss = self.VAE_loss(mse_loss, MU_Z_eval, LOG_VAR_Z_eval)

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            ELBO[epoch] = train_loss / train_loss_num
            if epoch % 10 == 0:
                log.info(
                    "[Epoch: {}/{}] [objective: {:.3f}]".format(
                        epoch, epochs, ELBO[epoch, 0]
                    )
                )

            ELBO_train = ELBO[epoch, 0].round(2)
            log.info("[ELBO train: " + str(ELBO_train) + "]")
        del MU_X_eval, MU_Z_eval, Z_ENC_eval
        del LOG_VAR_X_eval, LOG_VAR_Z_eval

        self.save()
        log.info("... finished training of Variational Autoencoder.")

    def load(self, input_shape):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "{}_{}.{}".format(self._data_name, input_shape, "pt"),
        )

        self.load_state_dict(torch.load(load_path))

        self.eval()

        return self

    def save(self):
        cache_path = get_home()

        save_path = os.path.join(
            cache_path,
            "{}_{}.{}".format(self._data_name, self._input_dim, "pt"),
        )

        torch.save(self.state_dict(), save_path)


class CSVAE(nn.Module):
    def __init__(self, data_name: str, layers: List[int]) -> None:
        super(CSVAE, self).__init__()
        self._input_dim = layers[0]
        self.z_dim = layers[-1]
        self._data_name = data_name
        # w_dim and labels_dim are fix due to our constraint to binary labeled data
        w_dim = 2
        self._labels_dim = w_dim

        # encoder
        lst_encoder_xy_to_w = []
        for i in range(1, len(layers) - 1):
            if i == 1:
                lst_encoder_xy_to_w.append(
                    nn.Linear(layers[i - 1] + self._labels_dim, layers[i])
                )
            else:
                lst_encoder_xy_to_w.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder_xy_to_w.append(nn.ReLU())
        self.encoder_xy_to_w = nn.Sequential(*lst_encoder_xy_to_w)

        self.mu_xy_to_w = nn.Sequential(
            self.encoder_xy_to_w, nn.Linear(layers[-2], w_dim)
        )

        self.logvar_xy_to_w = nn.Sequential(
            self.encoder_xy_to_w, nn.Linear(layers[-2], w_dim)
        )

        lst_encoder_x_to_z = copy.deepcopy(lst_encoder_xy_to_w)
        lst_encoder_x_to_z[0] = nn.Linear(layers[0], layers[1])
        self.encoder_x_to_z = nn.Sequential(*lst_encoder_x_to_z)

        self.mu_x_to_z = nn.Sequential(
            self.encoder_x_to_z, nn.Linear(layers[-2], self.z_dim)
        )

        self.logvar_x_to_z = nn.Sequential(
            self.encoder_x_to_z, nn.Linear(layers[-2], self.z_dim)
        )

        lst_encoder_y_to_w = copy.deepcopy(lst_encoder_xy_to_w)
        lst_encoder_y_to_w[0] = nn.Linear(self._labels_dim, layers[1])
        self.encoder_y_to_w = nn.Sequential(*lst_encoder_y_to_w)

        self.mu_y_to_w = nn.Sequential(
            self.encoder_y_to_w, nn.Linear(layers[-2], w_dim)
        )

        self.logvar_y_to_w = nn.Sequential(
            self.encoder_y_to_w, nn.Linear(layers[-2], w_dim)
        )

        # decoder
        lst_decoder_zw_to_x = []
        for i in range(len(layers) - 2, 0, -1):
            if i == len(layers) - 2:
                lst_decoder_zw_to_x.append(nn.Linear(layers[i + 1] + w_dim, layers[i]))
            else:
                lst_decoder_zw_to_x.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder_zw_to_x.append(nn.ReLU())
        self.decoder_zw_to_x = nn.Sequential(*lst_decoder_zw_to_x)

        self.mu_zw_to_x = nn.Sequential(
            self.decoder_zw_to_x, nn.Linear(layers[1], self._input_dim)
        )

        self.logvar_zw_to_x = nn.Sequential(
            self.decoder_zw_to_x, nn.Linear(layers[1], self._input_dim)
        )

        lst_decoder_z_to_y = copy.deepcopy(lst_decoder_zw_to_x)
        lst_decoder_z_to_y[0] = nn.Linear(self.z_dim, layers[-2])
        lst_decoder_z_to_y.append(nn.Linear(layers[1], self._labels_dim))
        lst_decoder_z_to_y.append(nn.Sigmoid())
        self.decoder_z_to_y = nn.Sequential(*lst_decoder_z_to_y)

    def q_zw(self, x, y):
        xy = torch.cat([x, y], dim=1)

        z_mu = self.mu_x_to_z(x)
        z_logvar = self.logvar_x_to_z(x)

        w_mu_encoder = self.mu_xy_to_w(xy)
        w_logvar_encoder = self.logvar_xy_to_w(xy)

        w_mu_prior = self.mu_y_to_w(y)
        w_logvar_prior = self.logvar_y_to_w(y)

        return (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def p_x(self, z, w):
        zw = torch.cat([z, w], dim=1)

        mu = self.mu_zw_to_x(zw)
        logvar = self.logvar_zw_to_x(zw)

        return mu, logvar

    def forward(self, x, y):
        (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        ) = self.q_zw(x, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        z = self.reparameterize(z_mu, z_logvar)
        zw = torch.cat([z, w_encoder], dim=1)

        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred = self.decoder_z_to_y(z)

        return (
            x_mu,
            x_logvar,
            zw,
            y_pred,
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def predict(self, x, y):
        return self.forward(x, y)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    def fit(self, data, lambda_reg=None, epochs=100, lr=1e-3, batch_size=32):
        x_train = data[:, :-1]

        if self._labels_dim == 2:
            y_prob_train = np.zeros((data.shape[0], 2))
            y_prob_train[:, 0] = 1 - data[:, -1]
            y_prob_train[:, 1] = data[:, -1]
        else:
            raise ValueError("Only binary class labels are implemented at the moment.")

        train_loader = torch.utils.data.DataLoader(
            list(zip(x_train, y_prob_train)), shuffle=True, batch_size=batch_size
        )

        params_without_delta = [
            param
            for name, param in self.named_parameters()
            if "decoder_z_to_y" not in name
        ]
        params_delta = [
            param for name, param in self.named_parameters() if "decoder_z_to_y" in name
        ]

        opt_without_delta = optim.Adam(params_without_delta, lr=lr / 2)
        opt_delta = optim.Adam(params_delta, lr=lr / 2)

        train_x_recon_losses = []
        train_y_recon_losses = []

        log.info("Start training of CSVAE...")
        for i in trange(epochs):
            for x, y in train_loader:
                (
                    loss_val,
                    x_recon_loss_val,
                    w_kl_loss_val,
                    z_kl_loss_val,
                    y_negentropy_loss_val,
                    y_recon_loss_val,
                ) = csvae_loss(self, x, y)

                opt_delta.zero_grad()
                y_recon_loss_val.backward(retain_graph=True)

                opt_without_delta.zero_grad()
                loss_val.backward()

                opt_without_delta.step()
                opt_delta.step()

                train_x_recon_losses.append(x_recon_loss_val.item())
                train_y_recon_losses.append(y_recon_loss_val.item())

            log.info(
                "epoch {}: x recon loss: {}".format(
                    i, np.mean(np.array(train_x_recon_losses))
                )
            )
            log.info(
                "epoch {}: y recon loss: {}".format(
                    i, np.mean(np.array(train_y_recon_losses))
                )
            )

        self.save()
        log.info("... finished training of CSVAE")

    def save(self):
        cache_path = get_home()

        save_path = os.path.join(
            cache_path,
            "csvae_{}_{}.{}".format(self._data_name, self._input_dim, "pt"),
        )

        torch.save(self.state_dict(), save_path)

    def load(self, input_shape):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "csvae_{}_{}.{}".format(self._data_name, input_shape, "pt"),
        )

        self.load_state_dict(torch.load(load_path))

        self.eval()

        return self
