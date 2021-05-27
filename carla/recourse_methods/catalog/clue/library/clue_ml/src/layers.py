from __future__ import division

from typing import Dict

import torch
import torch.nn as nn


def MLPBlock(width):
    return SkipConnection(
        nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)
    )


def preact_MLPBlock(width):
    return SkipConnection(
        nn.ReLU(),
        nn.BatchNorm1d(num_features=width),
        nn.Linear(width, width),
    )


def leaky_MLPBlock(width):
    return SkipConnection(
        nn.Linear(width, width), nn.LeakyReLU(), nn.BatchNorm1d(num_features=width)
    )


def preact_leaky_MLPBlock(width):
    return SkipConnection(
        nn.LeakyReLU(),
        nn.BatchNorm1d(num_features=width),
        nn.Linear(width, width),
    )


class ResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """

    def __init__(self, outer_dim, inner_dim):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)


class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """

    def __init__(self, *args):
        super(SkipConnection, self).__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)


class MemoryLayer(nn.Module):
    """
    If output=False, this layer stores its input in a static class dictionary
    `storage` with the key `id` and then passes the input to the next layer.
    If output=True, this layer takes stored tensor from a static storage.
    If add=True, it returns sum of the stored vector and an input,
    otherwise it returns their concatenation.
    If the tensor with specified `id` is not in `storage` when the layer
    with output=True is called, it would cause an exception.
    The layer is used to make skip-connections inside nn.Sequential network
    or between several nn.Sequential networks without unnecessary code
    complication.
    The usage pattern is
    ```
        net1 = nn.Sequential(
            MemoryLayer('#1'),
            MemoryLayer('#0.1'),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            MemoryLayer('#0.1', output=True, add=False),
            # here add cannot be True because the dimensions mismatch
            nn.Linear(768, 256),
            # the dimension after the concatenation with skip-connection
            # is 512 + 256 = 768
        )
        net2 = nn.Sequential(
            nn.Linear(512, 512),
            MemoryLayer('#1', output=True, add=True),
            ...
        )
        b = net1(a)
        d = net2(c)
        # net2 must be called after net1,
        # otherwise tensor '#1' will not be in `storage`
    ```
    """

    storage: Dict = {}

    def __init__(self, id, output=False, add=False):
        super(MemoryLayer, self).__init__()
        self.id = id
        self.output = output
        self.add = add

    def forward(self, input):
        if not self.output:
            self.storage[self.id] = input
            return input
        else:
            if self.id not in self.storage:
                err = "MemoryLayer: id '%s' is not initialized. "
                err += "You must execute MemoryLayer with the same id "
                err += "and output=False before this layer."
                raise ValueError(err)
            stored = self.storage[self.id]
            if not self.add:
                data = torch.cat([input, stored], 1)
            else:
                data = input + stored
            return data
