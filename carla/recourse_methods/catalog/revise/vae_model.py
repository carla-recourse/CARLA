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

    # Computes the objective function of the VAE
    # def VAE_loss(self, x, mu_x, log_var_x, mu_z, log_var_z, r=1.0):
    #     D = mu_x.shape[1]
    #     d = mu_z.shape[1]
    #
    #     if log_var_x.shape[1] == 1:
    #         P_X_Z = (
    #             +0.5
    #             * (
    #                 D * log_var_x
    #                 + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)
    #             ).mean()
    #         )
    #     else:
    #         P_X_Z = (
    #             +0.5
    #             * (
    #                 log_var_x.sum(dim=1, keepdim=True)
    #                 + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)
    #             ).mean()
    #         )
    #
    #     if log_var_z.shape[1] == 1:
    #         Q_Z_X = -0.5 * (d * log_var_z).mean()
    #     else:
    #         Q_Z_X = -0.5 * log_var_z.sum(dim=1, keepdim=True).mean()
    #
    #     if log_var_z.shape[1] == 1:
    #         P_Z = (
    #             +0.5
    #             * ((mu_z ** 2).sum(dim=1, keepdim=True) + d * log_var_z.exp()).mean()
    #         )
    #     else:
    #         P_Z = (
    #             +0.5
    #             * (
    #                 (mu_z ** 2).sum(dim=1, keepdim=True)
    #                 + log_var_z.exp().sum(dim=1, keepdim=True)
    #             ).mean()
    #         )
    #
    #     return P_X_Z + r * Q_Z_X + r * P_Z

    def VAE_loss(self, mse_loss, mu, logvar):
        MSE = mse_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    def load(self, pathname):
        self.load_state_dict(torch.load(pathname))
