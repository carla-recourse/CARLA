import copy
import os
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange

from carla import log
from carla.recourse_methods.autoencoder.losses import csvae_loss
from carla.recourse_methods.autoencoder.save_load import get_home

tf.compat.v1.disable_eager_execution()


class CSVAE(nn.Module):
    def __init__(self, data_name: str, layers: List[int], mutable_mask) -> None:
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
        # the decoder does use the immutables, so need to increase layer size accordingly.
        layers[-1] += np.sum(~mutable_mask)
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
        lst_decoder_z_to_y[0] = nn.Linear(
            self.z_dim + np.sum(~mutable_mask), layers[-2]
        )
        lst_decoder_z_to_y.append(nn.Linear(layers[1], self._labels_dim))
        lst_decoder_z_to_y.append(nn.Sigmoid())
        self.decoder_z_to_y = nn.Sequential(*lst_decoder_z_to_y)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.to(device)

        self.mutable_mask = mutable_mask

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

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]

        (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        ) = self.q_zw(x_mutable, y)

        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        z = self.reparameterize(z_mu, z_logvar)

        # concatenate the immutable part to the latents
        z = torch.cat([z, x_immutable], dim=-1)

        zw = torch.cat([z, w_encoder], dim=1)

        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred = self.decoder_z_to_y(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = x_mu
        x_mu = x

        # set variance to zero (one in log space) for immutable features
        temp = torch.ones_like(x)
        temp[:, self.mutable_mask] = x_logvar
        x_logvar = temp

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

    def fit(self, data, epochs=100, lr=1e-3, batch_size=32):
        if isinstance(data, pd.DataFrame):
            data = data.values
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
                x = x.to(self.device)
                y = y.to(self.device)
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

        self.eval()

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
