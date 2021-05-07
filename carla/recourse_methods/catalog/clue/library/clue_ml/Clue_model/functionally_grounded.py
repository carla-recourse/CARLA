from __future__ import division

import numpy as np
import torch
import torch.nn.functional as F
from interpret.generate_data import (
    sample_artificial_targets_bern,
    sample_artificial_targets_cat,
    sample_artificial_targets_gauss,
)
from src.gauss_cat import flat_to_gauss_cat, rms_cat_loglike
from src.probability import decompose_entropy_cat, decompose_std_gauss
from src.utils import MNIST_mean_std_norm, generate_ind_batch


def get_BNN_uncertainties(
    BNN,
    explanations,
    regression,
    batch_size=1024,
    norm_MNIST=False,
    flatten=False,
    return_probs=False,
    prob_BNN=True,
):
    total_stack = []
    aleatoric_stack = []
    epistemic_stack = []
    probs_stack = []
    aux_loader = generate_ind_batch(
        explanations.shape[0], batch_size=batch_size, random=False, roundup=True
    )
    for idxs in aux_loader:
        if regression:

            if prob_BNN:
                mu_vec, std_vec = BNN.sample_predict(
                    explanations[idxs], Nsamples=0, grad=False
                )
                (
                    total_uncertainty,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                ) = decompose_std_gauss(mu_vec, std_vec)
                probs_stack.append(mu_vec)
            else:
                mu, std = BNN.predict(explanations[idxs], grad=False)
                probs_stack.append(mu)
                total_uncertainty = std
                aleatoric_uncertainty = std
                epistemic_uncertainty = std * 0
        else:

            if norm_MNIST:
                to_BNN = MNIST_mean_std_norm(explanations[idxs])
            else:
                to_BNN = explanations[idxs]

            if flatten:
                to_BNN = to_BNN.view(to_BNN.shape[0], -1)

            if prob_BNN:
                probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False)
                (
                    total_uncertainty,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                ) = decompose_entropy_cat(probs)
                probs_stack.append(probs)
            else:
                probs = BNN.predict(to_BNN, grad=False)
                total_uncertainty = -(probs * torch.log(probs + 1e-10)).sum(
                    dim=1, keepdim=False
                )
                aleatoric_uncertainty = total_uncertainty
                epistemic_uncertainty = total_uncertainty * 0
                probs_stack.append(probs)

        total_stack.append(total_uncertainty)
        aleatoric_stack.append(aleatoric_uncertainty)
        epistemic_stack.append(epistemic_uncertainty)

    total_stack = torch.cat(total_stack, dim=0)
    aleatoric_stack = torch.cat(aleatoric_stack, dim=0)
    epistemic_stack = torch.cat(epistemic_stack, dim=0)
    probs_stack = torch.cat(probs_stack, dim=1)
    if return_probs:
        return total_stack, aleatoric_stack, epistemic_stack, probs_stack
    else:
        return total_stack, aleatoric_stack, epistemic_stack


def evaluate_aleatoric_explanation_cat(
    VAEAC, explanations, test_dims, N_target_samples=500, batch_size=1024
):
    """This assumes that test dims are placed at end of input vector"""
    explanations = explanations.view(explanations.shape[0], -1)
    explanations_expand = torch.cat(
        [explanations, explanations.new_zeros((explanations.shape[0], len(test_dims)))],
        dim=1,
    )
    assert explanations_expand.shape[1] == (test_dims[-1] + 1)

    output_stack = []
    aux_loader = generate_ind_batch(
        explanations_expand.shape[0], batch_size=batch_size, random=False, roundup=True
    )
    for idxs in aux_loader:
        probs = sample_artificial_targets_cat(
            VAEAC, explanations_expand[idxs], test_dims, N_target_samples
        ).data
        output_stack.append(probs.data.cpu())

    y_cond_explan_aleatoric_probs = torch.cat(output_stack, dim=1)
    y_cond_explan_aleatoric_entropy, _, _ = decompose_entropy_cat(
        y_cond_explan_aleatoric_probs
    )
    return y_cond_explan_aleatoric_entropy


def evaluate_aleatoric_explanation_MNIST(
    VAEAC, explanations, test_dims, N_target_samples=500, batch_size=1024
):
    """This assumes that test dims are placed at end of input vector"""
    explanations = explanations.view(explanations.shape[0], -1)
    explanations_expand = torch.cat(
        [explanations, explanations.new_zeros((explanations.shape[0], len(test_dims)))],
        dim=1,
    )
    assert explanations_expand.shape[1] == (test_dims[-1] + 1)

    output_stack = []
    aux_loader = generate_ind_batch(
        explanations_expand.shape[0], batch_size=batch_size, random=False, roundup=True
    )
    for idxs in aux_loader:
        probs = sample_artificial_targets_bern(
            VAEAC, explanations_expand[idxs], test_dims, N_target_samples
        ).data
        output_stack.append(probs.data.cpu())

    y_cond_explan_aleatoric_probs = torch.cat(output_stack, dim=1)
    y_cond_explan_aleatoric_entropy, _, _ = decompose_entropy_cat(
        y_cond_explan_aleatoric_probs
    )
    return y_cond_explan_aleatoric_entropy


def evaluate_aleatoric_explanation_gauss(
    VAEAC, explanations, test_dims, pred_sig, N_target_samples=500, batch_size=1024
):
    """This assumes that test dims are placed at end of input vector"""
    explanations = explanations.view(explanations.shape[0], -1)
    explanations_expand = torch.cat(
        [explanations, explanations.new_zeros((explanations.shape[0], len(test_dims)))],
        dim=1,
    )
    assert explanations_expand.shape[1] == (test_dims[-1] + 1)

    output_means_stack = []
    output_stds_stack = []
    aux_loader = generate_ind_batch(
        explanations_expand.shape[0], batch_size=batch_size, random=False, roundup=True
    )
    for idxs in aux_loader:
        means, stds = sample_artificial_targets_gauss(
            VAEAC,
            explanations_expand[idxs],
            test_dims,
            N_target_samples,
            pred_sig,
            z_mean=False,
        )
        output_means_stack.append(means.data.cpu())
        output_stds_stack.append(stds.data.cpu())

    y_cond_explan_aleatoric_means = torch.cat(output_means_stack, dim=1)
    y_cond_explan_aleatoric_stds = torch.cat(output_stds_stack, dim=1)
    y_cond_explan_aleatoric_entropy, _, _ = decompose_std_gauss(
        y_cond_explan_aleatoric_means, y_cond_explan_aleatoric_stds
    )
    return y_cond_explan_aleatoric_entropy


def evaluate_BNN_epistemic_class_error_loglike_cat(
    BNN, epistemic_explanations, explanation_targets, batch_size=1024, flatten=False
):
    if flatten:
        epistemic_explanations = epistemic_explanations.view(
            epistemic_explanations.shape[0], -1
        )

    aux_loader = generate_ind_batch(
        epistemic_explanations.shape[0], batch_size, random=False, roundup=True
    )
    loglike_vec = []
    test_err = 0
    nb_samples = 0
    for idxs in aux_loader:
        probs_samples = BNN.sample_predict(
            x=epistemic_explanations[idxs], Nsamples=0, grad=False
        ).data
        probs = probs_samples.mean(dim=0)

        log_probs = torch.log(probs)
        loss = F.nll_loss(log_probs, explanation_targets[idxs], reduction="none").data

        pred = probs.data.max(dim=1, keepdim=False)[
            1
        ]  # get the index of the max log-probability
        err = pred.ne(explanation_targets[idxs].data).sum()

        loglike_vec.append(-loss)
        test_err += err.cpu().numpy()
        nb_samples += len(idxs)

    test_err /= nb_samples
    loglike_vec = torch.cat(loglike_vec, dim=0)

    return test_err, loglike_vec


def evaluate_BNN_epistemic_err_gauss(
    BNN,
    epistemic_explanations,
    y_cond_explanations_mu_epistemic,
    batch_size=1000,
    flatten=False,
):
    # Nte that this function does not unnormalise
    if flatten:
        epistemic_explanations = epistemic_explanations.view(
            epistemic_explanations.shape[0], -1
        )

    aux_loader = generate_ind_batch(
        epistemic_explanations.shape[0], batch_size, random=False, roundup=True
    )
    diffs = []
    for idxs in aux_loader:
        mu, std = BNN.sample_predict(
            x=epistemic_explanations[idxs], Nsamples=0, grad=False
        )
        pred = mu.mean(dim=0).cpu()
        diffs.append((y_cond_explanations_mu_epistemic[idxs] - pred).abs().data)
    diffs = torch.cat(diffs, dim=0)

    rms = torch.sqrt((diffs ** 2).sum() / diffs.shape[0])

    return rms, diffs


def evaluate_epistemic_explanation_gauss(
    BNN,
    VAEAC,
    epistemic_explanation,
    test_dims,
    pred_sig,
    outer_batch_size=2000,
    inner_batch_size=1024,
    VAEAC_samples=500,
):
    epistemic_explanation = epistemic_explanation.view(
        epistemic_explanation.shape[0], -1
    )

    epistemic_explanation_expand = torch.cat(
        [
            epistemic_explanation,
            epistemic_explanation.new_zeros(
                epistemic_explanation.shape[0], len(test_dims)
            ),
        ],
        dim=1,
    )

    output_means_stack = []
    aux_loader = generate_ind_batch(
        epistemic_explanation_expand.shape[0],
        outer_batch_size,
        random=False,
        roundup=True,
    )
    for idxs in aux_loader:
        y_cond_x_mu, y_cond_x_std = sample_artificial_targets_gauss(
            VAEAC,
            epistemic_explanation_expand[idxs],
            pred_sig=pred_sig,
            test_dims=test_dims,
            N_target_samples=VAEAC_samples,
            z_mean=False,
        )
        output_means_stack.append(y_cond_x_mu.data.cpu())
    # Get the expected prediction from the BNN
    y_cond_explanation_epistemic_mu = torch.cat(output_means_stack, dim=1).mean(dim=0)

    rms, diffs = evaluate_BNN_epistemic_err_gauss(
        BNN,
        epistemic_explanation,
        y_cond_explanation_epistemic_mu,
        batch_size=inner_batch_size,
        flatten=False,
    )
    return rms.cpu().numpy(), diffs.cpu().numpy()


def evaluate_epistemic_explanation_cat(
    BNN,
    VAEAC,
    epistemic_explanation,
    test_dims,
    outer_batch_size=2000,
    inner_batch_size=1024,
    VAEAC_samples=500,
):
    epistemic_explanation = epistemic_explanation.view(
        epistemic_explanation.shape[0], -1
    )

    epistemic_explanation_expand = torch.cat(
        [
            epistemic_explanation,
            epistemic_explanation.new_zeros(
                epistemic_explanation.shape[0], len(test_dims)
            ),
        ],
        dim=1,
    )

    output_stack = []
    aux_loader = generate_ind_batch(
        epistemic_explanation_expand.shape[0],
        outer_batch_size,
        random=False,
        roundup=True,
    )
    for idxs in aux_loader:
        probs = sample_artificial_targets_cat(
            VAEAC,
            epistemic_explanation_expand[idxs],
            test_dims=test_dims,
            N_target_samples=VAEAC_samples,
            z_mean=False,
            softmax=False,
        ).data
        output_stack.append(probs.cpu())
    # We have the VAEAC give us its expected prediction
    y_cond_explanation_epistemic_probs = torch.cat(output_stack, dim=1).mean(dim=0)
    y_cond_explanation_epistemic_preds = y_cond_explanation_epistemic_probs.max(dim=1)[
        1
    ]

    test_err, loglike_vec = evaluate_BNN_epistemic_class_error_loglike_cat(
        BNN,
        epistemic_explanation,
        y_cond_explanation_epistemic_preds.cuda(),
        batch_size=inner_batch_size,
        flatten=False,
    )
    loglike_vec = loglike_vec.cpu().numpy()
    return test_err, loglike_vec


def evaluate_epistemic_explanation_MNIST(
    BNN,
    VAEAC,
    epistemic_explanation,
    test_dims,
    outer_batch_size=2000,
    inner_batch_size=1024,
    VAEAC_samples=500,
):
    epistemic_explanation = epistemic_explanation.view(
        epistemic_explanation.shape[0], -1
    )

    epistemic_explanation_expand = torch.cat(
        [
            epistemic_explanation,
            epistemic_explanation.new_zeros(
                epistemic_explanation.shape[0], len(test_dims)
            ),
        ],
        dim=1,
    )

    output_stack = []
    aux_loader = generate_ind_batch(
        epistemic_explanation_expand.shape[0],
        outer_batch_size,
        random=False,
        roundup=True,
    )
    for idxs in aux_loader:
        probs = sample_artificial_targets_bern(
            VAEAC,
            epistemic_explanation_expand[idxs],
            test_dims=test_dims,
            N_target_samples=VAEAC_samples,
            z_mean=False,
        ).data
        output_stack.append(probs.cpu())
    # We have the VAEAC give us its expected prediction
    y_cond_explanation_epistemic_probs = torch.cat(output_stack, dim=1).mean(dim=0)
    y_cond_explanation_epistemic_preds = y_cond_explanation_epistemic_probs.max(dim=1)[
        1
    ]

    epistemic_explanation_to_BNN = MNIST_mean_std_norm(epistemic_explanation)

    test_err, loglike_vec = evaluate_BNN_epistemic_class_error_loglike_cat(
        BNN,
        epistemic_explanation_to_BNN,
        y_cond_explanation_epistemic_preds.cuda(),
        batch_size=inner_batch_size,
        flatten=False,
    )
    loglike_vec = loglike_vec.cpu().numpy()
    return test_err, loglike_vec


def get_VAEAC_px(
    under_VAEAC_net, x_art_test, y_dims, Nsamples=5000, bern=False, batch_size=None
):
    """Note that this function automatically masks y and only takes x as input
    Works for factorised Bernouilli inputs and Gaussian inputs"""
    max_dims = x_art_test.shape[1] + len(y_dims)
    x_dims = range(max_dims)
    for e in y_dims:
        x_dims.remove(e)

    iw_xy = torch.zeros((x_art_test.shape[0], max_dims))
    iw_xy[:, x_dims] = torch.Tensor(x_art_test)
    iw_xy[:, y_dims] = iw_xy.new_zeros((iw_xy.shape[0], len(y_dims)))

    iw_mask = torch.zeros_like(iw_xy)
    iw_mask[:, y_dims] = 1

    prior = under_VAEAC_net.prior
    # batching from here
    log_px_vec = []

    if batch_size is None:
        batch_size = iw_xy.shape[0]
    aux_loader = generate_ind_batch(
        iw_xy.shape[0], batch_size, random=False, roundup=True
    )
    for idxs in aux_loader:

        u_approx_dist = under_VAEAC_net.u_mask_recongnition(
            iw_xy[idxs], iw_mask[idxs], grad=False
        )

        p_x_estimates = []
        for i in range(Nsamples):

            u_sample = u_approx_dist.sample().data
            rec_distrib = under_VAEAC_net.u_regenerate(u_sample, grad=False)

            log_p = prior.log_prob(u_sample).sum(dim=1).data
            log_q = u_approx_dist.log_prob(u_sample).sum(dim=1).data
            if bern:
                rec_loglike = -F.binary_cross_entropy_with_logits(
                    rec_distrib, iw_xy[idxs].type(rec_distrib.type()), reduction="none"
                )
            else:
                rec_loglike = rec_distrib.log_prob(
                    iw_xy[idxs].type(u_sample.type())
                ).data
            x_loglike = rec_loglike[:, x_dims].sum(dim=1)

            p_x_estimates.append(x_loglike + log_p - log_q)

        log_px = torch.logsumexp(
            torch.stack(p_x_estimates), dim=0, keepdim=False
        ) - np.log(Nsamples)

        log_px_vec.append(log_px)

    log_px_vec = torch.cat(log_px_vec, dim=0)
    return log_px_vec


def get_VAEAC_px_gauss_cat(
    under_VAEAC_net,
    x_art_test,
    input_dim_vec,
    y_dims,
    override_y_dims=None,
    Nsamples=5000,
):
    """Note that this function automatically masks y in generations and only takes x as input"""
    rec_loglike_func = rms_cat_loglike(input_dim_vec, reduction="none")
    max_dims = x_art_test.shape[1] + len(y_dims)
    x_dims = range(max_dims)
    for e in y_dims:
        x_dims.remove(e)

    iw_xy = torch.zeros((x_art_test.shape[0], max_dims))
    iw_xy[:, x_dims] = torch.Tensor(x_art_test)
    iw_xy[:, y_dims] = iw_xy.new_zeros(
        (iw_xy.shape[0], len(y_dims))
    ).normal_()  # this offsets for the max operation
    if under_VAEAC_net.cuda:
        iw_xy = iw_xy.cuda()

    iw_xy_target = flat_to_gauss_cat(iw_xy, input_dim_vec)

    iw_mask = torch.zeros_like(iw_xy)
    iw_mask[:, y_dims] = 1

    prior = under_VAEAC_net.prior
    u_approx_dist = under_VAEAC_net.u_mask_recongnition(iw_xy, iw_mask, grad=False)

    p_x_estimates = []
    for i in range(Nsamples):

        u_sample = u_approx_dist.sample()
        rec_distrib = under_VAEAC_net.u_regenerate(u_sample, grad=False)

        log_p = prior.log_prob(u_sample).sum(dim=1).data
        log_q = u_approx_dist.log_prob(u_sample).sum(dim=1).data
        rec_loglike = rec_loglike_func(rec_distrib, iw_xy_target).view(
            iw_xy.shape[0], -1
        )

        if override_y_dims is not None:
            x_loglike = rec_loglike[:, :-override_y_dims].sum(dim=1)
        else:
            x_loglike = rec_loglike[:, x_dims].sum(dim=1)
        p_x_estimates.append(x_loglike + log_p - log_q)

    log_px = torch.logsumexp(torch.stack(p_x_estimates), dim=0, keepdim=False) - np.log(
        Nsamples
    )
    return log_px
