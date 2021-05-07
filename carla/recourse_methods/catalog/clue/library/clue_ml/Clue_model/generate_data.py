from __future__ import division

import torch
import torch.nn.functional as F
from numpy.random import normal


def sample_artificial_dataset(
    under_VAEAC_net, test_dims, Npoints, u_dims, sig=True, bern=False, softmax=False
):

    u = torch.Tensor(normal(loc=0.0, scale=1.0, size=(Npoints, u_dims)))
    if sig:
        xy = under_VAEAC_net.u_regenerate(u).sample()
    else:
        xy = under_VAEAC_net.u_regenerate(u).data

    train_dims = list(range(xy.shape[1]))
    for e in test_dims:
        train_dims.remove(e)

    x_art = xy[:, train_dims]  # last is test
    if bern:
        x_art = torch.sigmoid(x_art)
    y_art = xy[:, test_dims]  # last is test
    if softmax:
        y_art = F.softmax(y_art, dim=1)

    xy[:, train_dims] = x_art
    xy[:, test_dims] = y_art

    return x_art, y_art, xy


def sample_artificial_targets_gauss(
    VAEAC_net, xy, test_dims, N_target_samples, pred_sig, z_mean=False
):

    y_mask = torch.zeros_like(xy)
    y_mask[:, test_dims] = 1  # values set to 1 are masked out
    # N_target_samples, Npoints
    y_cond_x = VAEAC_net.inpaint(xy, y_mask, Nsample=N_target_samples, z_mean=z_mean)
    if pred_sig:
        y_cond_x_mu = [i.loc[:, test_dims] for i in y_cond_x]
        y_cond_x_mu = torch.stack(y_cond_x_mu, dim=0)
        y_cond_x_std = [i.scale[:, test_dims] for i in y_cond_x]
        y_cond_x_std = torch.stack(y_cond_x_std, dim=0)
    else:
        y_cond_x_mu = y_cond_x[:, :, test_dims]
        y_cond_x_std = torch.ones(
            (y_cond_x.shape[0], y_cond_x.shape[1], len(test_dims))
        )
    return y_cond_x_mu, y_cond_x_std


def sample_artificial_targets_bern(
    VAEAC_net, xy, test_dims, N_target_samples, z_mean=False
):
    y_mask = torch.zeros_like(xy)
    y_mask[:, test_dims] = 1  # values set to 1 are masked out
    # N_target_samples, Npoints
    xy_cond_x = VAEAC_net.inpaint(
        xy, y_mask, Nsample=N_target_samples, z_mean=z_mean, logits=True
    ).data

    y_cond_x = xy_cond_x[:, :, test_dims]
    y_cond_x = F.softmax(y_cond_x, dim=2)

    return y_cond_x


def sample_artificial_targets_cat(
    VAEAC_net, xy, test_dims, N_target_samples, z_mean=False, softmax=False
):
    y_mask = torch.zeros_like(xy)
    y_mask[:, test_dims] = 1  # values set to 1 are masked out
    # N_target_samples, Npoints
    xy_cond_x = VAEAC_net.inpaint(
        xy, y_mask, Nsample=N_target_samples, z_mean=z_mean, cat_probs=True
    ).data

    y_cond_x = xy_cond_x[:, :, test_dims]
    if softmax:
        y_cond_x = F.softmax(y_cond_x, dim=2)

    return y_cond_x
