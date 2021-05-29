from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from carla.recourse_methods.catalog.clue.library.clue_ml.src.utils import torch_onehot


def selective_softmax(
    x, input_dim_vec, grad=False, cat_probs=False, prob_sample=False, eps=1e-20
):
    """Applies softmax operation to specified dimensions. Gradient estimator is optional.
    cat_probs returns probability vectors over categorical variables instead of maxing
    if cat_probs is activated with prob sample, a one-hot vector will be sampled (reparametrisable)"""
    output = torch.zeros_like(x)
    cum_dims = 0
    for idx, dim in enumerate(input_dim_vec):
        if dim == 1:
            output[:, cum_dims] = x[:, cum_dims]
            if prob_sample:  # this assumes an rms loss when training
                noise = x.new_zeros(x.shape[0]).normal_(mean=0, std=1)
                output[:, cum_dims] = output[:, cum_dims] + noise
            cum_dims += 1
        elif dim > 1:
            if not cat_probs:
                if not grad:
                    y = x[:, cum_dims : cum_dims + dim].max(dim=1)[1]
                    y_vec = torch_onehot(y, dim).type(x.type())
                    output[:, cum_dims : cum_dims + dim] = y_vec
                else:
                    x_cat = x[:, cum_dims : cum_dims + dim]
                    probs = F.softmax(x_cat, dim=1)
                    y_hard = x[:, cum_dims : cum_dims + dim].max(dim=1)[1]
                    y_oh = torch_onehot(y_hard, dim).type(x.type())
                    output[:, cum_dims : cum_dims + dim] = (
                        y_oh - probs
                    ).detach() + probs
            else:
                x_cat = x[:, cum_dims : cum_dims + dim]
                probs = F.softmax(x_cat, dim=1)

                if prob_sample:  # we are going to use gumbel trick here
                    log_probs = torch.log(probs)
                    u = log_probs.new(log_probs.shape).uniform_(0, 1)
                    g = -torch.log(-torch.log(u + eps) + eps)
                    cat_samples = (log_probs + g).max(dim=1)[1]
                    hard_samples = torch_onehot(cat_samples, dim).type(x.type())
                    output[:, cum_dims : cum_dims + dim] = hard_samples
                else:
                    output[:, cum_dims : cum_dims + dim] = probs

            cum_dims += dim
        else:
            raise ValueError("Error, invalid dimension value")
    return output


def gumbel_softmax(log_prob_map, temperature, eps=1e-20):
    """Note that inputs are logprobs"""
    u = log_prob_map.new(log_prob_map.shape).uniform_(0, 1)
    g = -torch.log(-torch.log(u + eps) + eps)
    softmax_in = (log_prob_map + eps) + g
    y = F.softmax(softmax_in / temperature, dim=1)
    y_hard = torch.max(y, dim=1)[1]
    y_hard = torch_onehot(y_hard, y.shape[1]).type(y.type())

    return (y_hard - y).detach() + y


def gauss_cat_to_flat(x, input_dim_vec):
    output = []
    for idx, dim in enumerate(input_dim_vec):
        if dim == 1:
            output.append(x[:, idx].unsqueeze(1))
        elif dim > 1:
            oh_vec = torch_onehot(x[:, idx], dim).type(x.type())
            output.append(oh_vec)
        else:
            raise ValueError("Error, invalid dimension value")
    return torch.cat(output, dim=1)


def gauss_cat_to_flat_mask(x, input_dim_vec):
    output = []
    for idx, dim in enumerate(input_dim_vec):
        if dim == 1:
            output.append(x[:, idx].unsqueeze(1))
        elif dim > 1:
            oh_vec = x.new_ones(x.shape[0], dim) * x[:, idx].unsqueeze(1)
            output.append(oh_vec)
        else:
            raise ValueError("Error, invalid dimension value")
    return torch.cat(output, dim=1)


def flat_to_gauss_cat(x, input_dim_vec):
    output = []
    cum_dims = 0
    for idx, dims in enumerate(input_dim_vec):
        if dims == 1:
            output.append(x[:, cum_dims].unsqueeze(1))
            cum_dims += 1

        elif dims > 1:
            output.append(
                x[:, cum_dims : cum_dims + dims]
                .max(dim=1)[1]
                .type(x.type())
                .unsqueeze(1)
            )
            cum_dims += dims

        else:
            raise ValueError("Error, invalid dimension value")

    return torch.cat(output, dim=1)


class rms_cat_loglike(nn.Module):
    def __init__(self, input_dim_vec, reduction="none"):
        super(rms_cat_loglike, self).__init__()
        self.reduction = reduction
        self.input_dim_vec = input_dim_vec
        self.mse = MSELoss(reduction="none")  # takes(input, target)
        self.ce = CrossEntropyLoss(reduction="none")

    def forward(self, x, y):

        log_prob_vec = []
        cum_dims = 0
        for idx, dims in enumerate(self.input_dim_vec):
            if dims == 1:
                # Gaussian_case
                log_prob_vec.append(-self.mse(x[:, cum_dims], y[:, idx]).unsqueeze(1))
                cum_dims += 1

            elif dims > 1:
                if x.shape[1] == y.shape[1]:
                    raise Exception(
                        "Input and target seem to be in flat format. Need integer cat targets."
                    )

                if y.is_cuda:
                    tget = y[:, idx].type(torch.cuda.LongTensor)
                else:
                    tget = y[:, idx].type(torch.LongTensor)

                log_prob_vec.append(
                    -self.ce(x[:, cum_dims : cum_dims + dims], tget).unsqueeze(1)
                )
                cum_dims += dims

            else:
                raise ValueError("Error, invalid dimension value")

        log_prob_vec = torch.cat(log_prob_vec, dim=1)

        if self.reduction == "none":
            return log_prob_vec
        elif self.reduction == "sum":
            return log_prob_vec.sum()
        elif self.reduction == "average":
            return log_prob_vec.mean()


# TODO: Implement for mu, sig inputs
class gauss_cat_loss(nn.Module):
    pass
