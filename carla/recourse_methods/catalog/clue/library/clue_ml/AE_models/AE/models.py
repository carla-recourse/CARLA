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


######################################
# Models for MNIST at 28x28
######################################
# ResNet MNIST conv model


class MNIST_recognition_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(MNIST_recognition_resnet, self).__init__()

        width_mul = 3

        proposal_layers = [
            nn.Conv2d(1, 32, kernel_size=1, padding=0, stride=1),  # 28x28 --32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 14x14 --64
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 7x7 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),  # 4x4 --128
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            nn.Linear(4 * 4 * 128, latent_dim * 2),
            SkipConnection(
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=latent_dim * 2),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            ),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MNIST_generator_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(MNIST_generator_resnet, self).__init__()

        width_mul = 3

        generative_layers = [
            leaky_MLPBlock(latent_dim),
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            small_MNIST_unFlatten(),  # 4x4 --128
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, padding=1, stride=2
            ),  # 7x7 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, stride=2
            ),  # 14x14 --64
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, stride=2
            ),  # 28x28 --32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1),
        ]  # 28x28 --1

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)  # ResNet MNIST conv model


# Small MNIST conv model


class small_MNIST_Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class small_MNIST_unFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 128, 4, 4)


class small_MNIST_recognition_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNIST_recognition_net, self).__init__()
        self.Nfilt1 = 256
        self.Nfilt2 = 128
        self.Nfilt3 = 128

        proposal_layers = [
            nn.Conv2d(1, self.Nfilt1, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt1),
            nn.Conv2d(self.Nfilt1, self.Nfilt2, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt2),
            nn.Conv2d(self.Nfilt2, self.Nfilt3, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=4 * 4 * self.Nfilt3),
            nn.Linear(4 * 4 * self.Nfilt3, latent_dim * 2),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class small_MNIST_generator_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNIST_generator_net, self).__init__()
        self.Nfilt1 = 256
        self.Nfilt2 = 128
        self.Nfilt3 = 128
        # input layer
        generative_layers = [
            nn.Linear(latent_dim, 4 * 4 * self.Nfilt3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=4 * 4 * self.Nfilt3),
            small_MNIST_unFlatten(),
            nn.ConvTranspose2d(
                self.Nfilt3, self.Nfilt2, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt3),
            nn.ConvTranspose2d(
                self.Nfilt2, self.Nfilt1, kernel_size=4, padding=1, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt1),
            nn.ConvTranspose2d(self.Nfilt1, 1, kernel_size=4, padding=1, stride=2),
        ]

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


######################################
# Models for Doodle at 64x64
######################################


class doodle_recognition_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(doodle_recognition_resnet, self).__init__()

        width_mul = 3

        proposal_layers = [
            nn.Conv2d(1, 32, kernel_size=1, padding=0, stride=1),  # 64x64 --32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 32x32 --64
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 16x16 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),  # 8x8 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),  # 4x4 --128
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            nn.Linear(4 * 4 * 128, latent_dim * 2),
            SkipConnection(
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=latent_dim * 2),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            ),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class doodle_generator_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(doodle_generator_resnet, self).__init__()

        width_mul = 3

        generative_layers = [
            leaky_MLPBlock(latent_dim),
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            small_MNIST_unFlatten(),  # 4x4 --128
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, padding=1, stride=2
            ),  # 8x8 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, padding=1, stride=2
            ),  # 16x16 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, stride=2
            ),  # 32x32 --64
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, stride=2
            ),  # 64x64 --32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1),
        ]  # 64x64 --1

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)  # ResNet MNIST conv model
