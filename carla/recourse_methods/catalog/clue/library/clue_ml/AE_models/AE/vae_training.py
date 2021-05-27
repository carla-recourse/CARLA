from __future__ import division

import torch

from carla.recourse_methods.catalog.clue.library.clue_ml.AE_models.AE.fc_gauss_cat import (
    VAE_gauss_cat_net,
)
from carla.recourse_methods.catalog.clue.library.clue_ml.AE_models.AE.train import (
    train_VAE,
)
from carla.recourse_methods.catalog.clue.library.clue_ml.src.utils import Datafeed


def training(
    x_train,
    x_test,
    input_dim_vec,
    path,
    width=10,
    depth=2,
    latent_dim=6,
    batch_size=128,
    nb_epochs=10,
    lr=1e-3,
    early_stop=25,
):
    """
    :param x_train: np.array; train set
    :param x_test: np.array; test set
    :param input_dim_vec: list; with dimensions of inputs (e.g. [1,2,1] for inputs ['#credit cards', 'sex' ,'age'])
    :param path: str; data set name (e.g. compas)
    :param width: int > 0; layer width
    :param depth: int > 0; layer depth
    :param latent_dim: int > 0; latent dim of VAE
    :param batch_size: int > 0; batch size
    :param nb_epochs: int > 0; # epochs
    :param lr: 0 < float < 1: learning rate
    :param early_stop: 0 < int < nb_epochs
    trains VAE and saves results
    """  #

    trainset = Datafeed(x_train, x_train, transform=None)
    valset = Datafeed(x_test, x_test, transform=None)

    # check whether GPU access
    cuda = torch.cuda.is_available()

    # load model architecture
    net = VAE_gauss_cat_net(
        input_dim_vec,
        width,
        depth,
        latent_dim,
        pred_sig=False,
        lr=lr,
        cuda=cuda,
        flatten=False,
    )

    # train & save model
    vlb_train, vlb_dev = train_VAE(
        net,
        path,
        batch_size,
        nb_epochs,
        trainset,
        valset,
        cuda=cuda,
        flat_ims=False,
        train_plot=False,
        early_stop=early_stop,
    )
