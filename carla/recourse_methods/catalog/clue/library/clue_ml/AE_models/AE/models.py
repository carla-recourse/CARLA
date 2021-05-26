from __future__ import division

import torch.nn as nn

from carla.recourse_methods.catalog.clue.library.clue_ml.src.layers import (
    MLPBlock,
    ResBlock,
    SkipConnection,
    leaky_MLPBlock,
    preact_leaky_MLPBlock,
)

# MLP based model


class MLP_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_recognition_net, self).__init__()
        # input layer
        proposal_layers = [
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        # body
        for i in range(depth - 1):
            proposal_layers.append(MLPBlock(width))
        # output layer
        proposal_layers.append(nn.Linear(width, latent_dim * 2))

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_generator_net, self).__init__()
        # input layer
        generative_layers = [
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        # body
        for i in range(depth - 1):
            generative_layers.append(
                # skip-connection from prior network to generative network
                leaky_MLPBlock(width)
            )
        # output layer
        generative_layers.extend(
            [
                nn.Linear(width, input_dim),
            ]
        )
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


# MLP fully linear residual path preact models


class MLP_preact_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_recognition_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim, width)]
        # body
        for i in range(depth - 1):
            proposal_layers.append(preact_leaky_MLPBlock(width))
        # output layer
        proposal_layers.extend(
            [
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=width),
                nn.Linear(width, latent_dim * 2),
            ]
        )

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_preact_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_generator_net, self).__init__()
        # input layer
        generative_layers = [
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        # body
        for i in range(depth - 1):
            generative_layers.append(
                # skip-connection from prior network to generative network
                preact_leaky_MLPBlock(width)
            )
        # output layer
        generative_layers.extend(
            [
                nn.Linear(width, input_dim),
            ]
        )
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)
