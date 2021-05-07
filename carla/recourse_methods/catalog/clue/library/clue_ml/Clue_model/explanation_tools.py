from __future__ import division

import torch
import torch.nn.functional as F
from interpret.generate_data import sample_artificial_targets_gauss
from src.probability import (
    decompose_entropy_cat,
    decompose_std_gauss,
    get_rms,
    marginal_std,
)
from src.utils import *
from torch.nn.functional import l1_loss

# TODO: Finish deprecating rest of this file and convert it to a sensitivity file


def sensitivity_analysis_cat(
    BNN, dset, aleatoric_coeff, epistemic_coeff, batch_size=1024, cuda=True
):
    """Returns mean of gradient absoluteb values at each input"""
    if cuda:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3
        )

    else:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )

    sensitivity = None
    total_sample = 0
    for x, y in trainloader:
        (x,) = to_variable(var=(x,), cuda=cuda)

        probs = BNN.sample_predict(x, Nsamples=0, grad=True)

        total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(
            probs
        )

        objective = (
            aleatoric_coeff * aleatoric_entropy.sum()
            + epistemic_coeff * epistemic_entropy.sum()
        )
        objective.backward()

        if sensitivity is None:
            sensitivity = torch.zeros(x.shape[1])
            if cuda:
                sensitivity = sensitivity.cuda()
        sensitivity += torch.abs(x.grad).sum(dim=0)
        total_sample += len(x)
    sensitivity = sensitivity / total_sample
    return sensitivity


def sensitivity_analysis_gauss(
    BNN,
    dset,
    aleatoric_coeff,
    epistemic_coeff,
    batch_size=1024,
    cuda=True,
    entropy=False,
):
    """Returns mean of gradient absoluteb values at each input"""
    if cuda:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3
        )

    else:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )

    sensitivity = None
    sensitivity_vec = []

    total_sample = 0
    for x, y in trainloader:
        (x,) = to_variable(var=(x,), cuda=cuda)

        mu_vec, std_vec = BNN.sample_predict(x, Nsamples=0, grad=True)

        if entropy:
            raise Exception("Deprecated option, will remove soon")
            # total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_N_gauss(mu_vec, std_vec)
        else:
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_std_gauss(
                mu_vec, std_vec
            )

        objective = (
            aleatoric_coeff * aleatoric_entropy.sum()
            + epistemic_coeff * epistemic_entropy.sum()
        )
        objective.backward()

        if sensitivity is None:
            sensitivity = torch.zeros(x.shape[1])
            if cuda:
                sensitivity = sensitivity.cuda()
        sensitivity += torch.abs(x.grad).sum(dim=0)
        sensitivity_vec.append(torch.abs(x.grad))
        total_sample += len(x)

    sensitivity = sensitivity / total_sample
    sensitivity_vec = torch.stack(sensitivity_vec)

    return sensitivity, sensitivity_vec


def input_uncertainty_step_gauss(
    BNN,
    dset,
    aleatoric_coeff,
    epistemic_coeff,
    stepsize_perdim=-1,
    batch_size=1024,
    cuda=True,
    entropy=False,
    norm_grad=False,
):
    """Takes a single step in the direction of uncertainty gradient wrt input"""
    if cuda:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )
    output_x = []
    for x, y in trainloader:
        (x,) = to_variable(var=(x,), cuda=cuda)

        mu_vec, std_vec = BNN.sample_predict(x, Nsamples=0, grad=True)

        if entropy:
            raise Exception("Deprecated option, will remove soon")
            # total_uncert, aleatoric_uncert, epistemic_uncert = decompose_entropy_N_gauss(mu_vec, std_vec)
        else:
            total_uncert, aleatoric_uncert, epistemic_uncert = decompose_std_gauss(
                mu_vec, std_vec
            )

        objective = (
            aleatoric_coeff * aleatoric_uncert.sum()
            + epistemic_coeff * epistemic_uncert.sum()
        )
        objective.backward()

        if norm_grad:
            l1_norm_step_dir = x.grad / (
                torch.abs(x.grad).sum(dim=1, keepdim=True) / x.shape[1] + 1e-12
            )
        else:
            l1_norm_step_dir = x.grad

        new_x = (
            x + stepsize_perdim * l1_norm_step_dir
        )  # gradient descent is induced by negative coeff in function input
        output_x.append(new_x)

    output_x = torch.cat(output_x)
    return output_x


def input_uncertainty_step_cat(
    BNN,
    dset,
    aleatoric_coeff,
    epistemic_coeff,
    stepsize_perdim=-1,
    batch_size=1024,
    cuda=True,
    norm_MNIST=False,
    flatten=False,
    norm_grad=False,
):
    """Takes a single step in the direction of uncertainty gradient wrt input"""
    if cuda:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )
    output_x = []
    for x, y in trainloader:
        (x,) = to_variable(var=(x,), cuda=cuda)
        x.requires_grad = True

        if norm_MNIST:
            to_BNN = MNIST_mean_std_norm(x)
        else:
            to_BNN = x

        if flatten:
            to_BNN = to_BNN.view(to_BNN.shape[0], -1)

        probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
        _, aleatoric_uncert, epistemic_uncert = decompose_entropy_cat(probs)
        objective = (
            aleatoric_coeff * aleatoric_uncert.sum()
            + epistemic_coeff * epistemic_uncert.sum()
        )
        objective.backward()

        if norm_grad:
            l1_norm_step_dir = x.grad / (
                torch.abs(x.grad).sum(dim=1, keepdim=True) / x.shape[1] + 1e-12
            )
        else:
            l1_norm_step_dir = x.grad

        new_x = (
            x + stepsize_perdim * l1_norm_step_dir
        )  # gradient descent is induced by negative coeffs
        output_x.append(new_x)

    output_x = torch.cat(output_x)
    return output_x


def evaluate_aleatoric_std(VAEAC, explanations, test_dims, N_target_samples=500):
    """This assumes that test dims are placed at end of input vector"""
    explanations_expand = torch.cat(
        [explanations, explanations.new_zeros((explanations.shape[0], len(test_dims)))],
        dim=1,
    )

    y_cond_explanations_mu, y_cond_explanations_std = sample_artificial_targets_gauss(
        VAEAC, explanations_expand, test_dims, N_target_samples
    )

    y_cond_marg_std = marginal_std(y_cond_explanations_mu, y_cond_explanations_std)
    y_cond_marg_std = y_cond_marg_std.cpu().numpy()

    return y_cond_marg_std


def evaluate_epistemic_rms(BNN, VAEAC, explanations, test_dims, nsamples=500):
    """This assumes that test dims are placed at end of input vector"""
    explanations_expand = torch.cat(
        [explanations, explanations.new_zeros((explanations.shape[0], len(test_dims)))],
        dim=1,
    )
    y_cond_explanations_mu_epistemic, _ = sample_artificial_targets_gauss(
        VAEAC, explanations_expand, test_dims, N_target_samples=nsamples, z_mean=False
    )

    y_cond_explanations_mu_epistemic = y_cond_explanations_mu_epistemic.mean(
        dim=0
    )  # we have 1 sample

    sensitivity_rms, x_all_abs_error = evaluate_BNN_epistemic_rms(
        BNN, explanations, y_cond_explanations_mu_epistemic
    )

    x_all_abs_error = x_all_abs_error.cpu().numpy()

    return sensitivity_rms, x_all_abs_error


def evaluate_BNN_epistemic_rms(BNN, epistemic_explanations, explanation_targets):
    mu_vec, std_vec = BNN.sample_predict(x=epistemic_explanations, Nsamples=0)
    rms = get_rms(mu_vec.mean(dim=0), y=explanation_targets, y_means=1, y_stds=1)
    all_abs_error = torch.abs(mu_vec.mean(dim=0) - explanation_targets)
    return rms, all_abs_error
