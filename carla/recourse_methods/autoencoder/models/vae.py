import os
from typing import List

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from carla import log
from carla.recourse_methods.autoencoder.dataloader import VAEDataset
from carla.recourse_methods.autoencoder.save_load import get_home

tf.compat.v1.disable_eager_execution()


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
        train_set = VAEDataset(xtrain, with_target=True)

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

        self.eval()

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
