# flake8: noqa
from __future__ import division

import numpy as np
import torch
from torch.distributions import Categorical, constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import StickBreakingTransform
from torch.nn import Module
from torch.nn.functional import softmax, softplus


# TODO: this might be better changed to a softmax transform instead of stick breaking but would need to look into it
class LogisticNormal(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `StickBreakingTransform` such that::
        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)
    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution
    Example::
        >>> # logistic-normal distributed with mean=(0, 0, 0) and stddev=(1, 1, 1)
        >>> # of the base Normal distribution
        >>> m = distributions.LogisticNormal(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))
        >>> m.sample()
        tensor([ 0.7653,  0.0341,  0.0579,  0.1427])
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        super(LogisticNormal, self).__init__(
            base_dist, StickBreakingTransform(), validate_args=validate_args
        )
        # Adjust event shape since StickBreakingTransform adds 1 dimension
        self._event_shape = torch.Size([s + 1 for s in self._event_shape])

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticNormal, _instance)
        return super(LogisticNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale


def normal_parse_params(params, min_sigma=1e-3):
    """
    Take a Tensor (e. g. neural network output) and return
    torch.distributions.Normal distribution.
    This Normal distribution is component-wise independent,
    and its dimensionality depends on the input shape.
    First half of channels is mean of the distribution,
    the softplus of the second half is std (sigma), so there is
    no restrictions on the input tensor.
    min_sigma is the minimal value of sigma. I. e. if the above
    softplus is less than min_sigma, then sigma is clipped
    from below with value min_sigma. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    n = params.shape[0]
    d = params.shape[1]
    mu = params[:, : d // 2]
    sigma_params = params[:, d // 2 :]
    sigma = softplus(sigma_params)
    sigma = sigma.clamp(min=min_sigma)
    distr = Normal(mu, sigma)
    return distr


def categorical_parse_params_column(params, min_prob=1e-2):
    """
    Take a Tensor (e. g. a part of neural network output) and return
    torch.distributions.Categorical distribution.
    The input tensor after applying softmax over the last axis contains
    a batch of the categorical probabilities. So there are no restrictions
    on the input tensor.
    Technically, this function treats the last axis as the categorical
    probabilities, but Categorical takes only 2D input where
    the first axis is the batch axis and the second one corresponds
    to the probabilities, so practically the function requires 2D input
    with the batch of probabilities for one categorical feature.
    min_prob is the minimal probability for each class.
    After clipping the probabilities from below they are renormalized
    in order to be a valid distribution. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    params = softmax(params, -1)
    params = params.clamp(min_prob)
    params = params / params.sum(-1, keepdim=True)
    distr = Categorical(probs=params)
    return distr


class GaussianLoglike(Module):
    """
    Compute reconstruction log probability of groundtruth given
    a tensor of Gaussian distribution parameters and a mask.
    Gaussian distribution parameters are output of a neural network
    without any restrictions, the minimal sigma value is clipped
    from below to min_sigma (default: 1e-2) in order not to overfit
    network on some exact pixels.
    The first half of channels corresponds to mean, the second half
    corresponds to std. See normal_parse_parameters for more info.
    This layer doesn't work with NaNs in the data, it is used for
    inpainting. Roughly speaking, this loss is similar to L2 loss.
    Returns a vector of log probabilities for each object of the batch.
    """

    def __init__(self, min_sigma=1e-2):
        super(GaussianLoglike, self).__init__()
        self.min_sigma = min_sigma

    def forward(self, distr_params, groundtruth, mask=None):
        distr = normal_parse_params(distr_params, self.min_sigma)
        if mask is not None:
            log_probs = distr.log_prob(groundtruth) * mask
        else:
            log_probs = distr.log_prob(groundtruth)
        return log_probs.view(groundtruth.shape[0], -1).sum(-1)


# functions for BNN with gauss output:
def diagonal_gauss_loglike(x, mu, sigma):
    # note that we can just treat each dim as isotropic and then do sum
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)
    log_px = (cte_term + det_sig_term + dist_term).sum(dim=1, keepdim=False)
    return log_px


def get_rms(mu, y, y_means, y_stds):
    x_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    return torch.sqrt(((x_un - y_un) ** 2).sum() / y.shape[0])


def gmm_likelihood(x, mu_vec, sigma_vec):
    weight_factor = np.log(mu_vec.shape[0])
    loglike_terms = []
    for i in range(mu_vec.shape[0]):
        loglike_terms.append(diagonal_gauss_loglike(x, mu_vec[i], sigma_vec[i]))
    loglike_terms = torch.cat(loglike_terms, dim=0)

    out = torch.logsumexp(loglike_terms, dim=0) - weight_factor
    return out


# TODO: rename this function to something gaussian
def get_loglike(mu, sigma, y, y_means, y_stds, gmm=False):
    mu_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    sigma_un = sigma * y_stds
    if gmm:
        ll = gmm_likelihood(y_un, mu_un, sigma_un)
    else:
        ll = diagonal_gauss_loglike(y_un, mu_un, sigma_un)
    return ll.mean(dim=0)  # mean over datapoints


def complete_logit_norm_vec(vec):
    last_term = 1 - vec.sum(dim=1)
    cvec = torch.cat((vec, last_term), dim=1)
    return cvec


def marginal_std(mu, sigma):  # This is for outputs from NN and GMM var estimation
    """Obtain the std of a GMM with isotropic components"""
    # probs (Nsamples, batch_size, classes)
    marg_var = (sigma ** 2).mean(dim=0) + ((mu ** 2).mean(dim=0) - mu.mean(dim=0) ** 2)
    return torch.sqrt(marg_var)


total_std = marginal_std


def decompose_var_gauss(mu, sigma, sum_dims=True):
    # probs (Nsamples, batch_size, output_sims)
    aleatoric_var = (sigma ** 2).mean(dim=0)
    epistemic_var = (mu ** 2).mean(dim=0) - mu.mean(dim=0) ** 2
    total_var = aleatoric_var + epistemic_var
    if sum_dims:
        aleatoric_var = aleatoric_var.sum(dim=1)
        epistemic_var = epistemic_var.sum(dim=1)
        total_var = total_var.sum(dim=1)
    return total_var, aleatoric_var, epistemic_var


def decompose_std_gauss(mu, sigma, sum_dims=True):
    # probs (Nsamples, batch_size, output_sims)
    aleatoric_var = (sigma ** 2).mean(dim=0)
    epistemic_var = (mu ** 2).mean(dim=0) - mu.mean(dim=0) ** 2
    total_var = aleatoric_var + epistemic_var
    if sum_dims:
        aleatoric_var = aleatoric_var.sum(dim=1)
        epistemic_var = epistemic_var.sum(dim=1)
        total_var = total_var.sum(dim=1)
    return total_var.sqrt(), aleatoric_var.sqrt(), epistemic_var.sqrt()


def decompose_entropy_cat(probs, eps=1e-10):
    # probs (Nsamples, batch_size, classes)
    posterior_preds = probs.mean(dim=0, keepdim=False)
    total_entropy = -(posterior_preds * torch.log(posterior_preds + eps)).sum(
        dim=1, keepdim=False
    )

    sample_preds_entropy = -(probs * torch.log(probs + eps)).sum(dim=2, keepdim=False)
    aleatoric_entropy = sample_preds_entropy.mean(dim=0, keepdim=False)

    epistemic_entropy = total_entropy - aleatoric_entropy

    # returns (batch_size)
    return total_entropy, aleatoric_entropy, epistemic_entropy
