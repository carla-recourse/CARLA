# flake8: noqa
import torch
import torch.nn as nn


class VAE_model(nn.Module):
    def __init__(self, latent_dim, input_dim, H1, H2, activFun):
        super(VAE_model, self).__init__()

        # The VAE components
        self.enc = nn.Sequential(
            nn.Linear(input_dim, H1),
            nn.BatchNorm1d(H1),
            activFun,
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            activFun,
        )

        self.mu_enc = nn.Sequential(self.enc, nn.Linear(H2, latent_dim))

        self.log_var_enc = nn.Sequential(self.enc, nn.Linear(H2, latent_dim))

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, H2),
            nn.BatchNorm1d(H2),
            activFun,
            nn.Linear(H2, H1),
            nn.BatchNorm1d(H1),
            activFun,
        )

        self.mu_dec = nn.Sequential(
            self.dec, nn.Linear(H1, input_dim), nn.BatchNorm1d(input_dim), nn.Sigmoid()
        )

        self.log_var_dec = nn.Sequential(
            self.dec, nn.Linear(H1, input_dim), nn.BatchNorm1d(input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        return self.mu_enc(x), self.log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)

    @staticmethod
    def reparametrization_trick(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z_rep = self.reparametrization_trick(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z_rep)

        return mu_x, log_var_x, z_rep, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)

    def regenerate(self, z, grad=False):
        mu_x, log_var_x = self.decode(z)
        return mu_x

    def VAE_loss(self, mse_loss, mu, logvar):
        MSE = mse_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    def load(self, pathname):
        self.load_state_dict(torch.load(pathname))
