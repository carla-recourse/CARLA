from __future__ import division

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from src.layers import SkipConnection
from src.probability import GaussianLoglike, normal_parse_params
from src.radam import RAdam
from src.utils import BaseNet, cprint, to_variable
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import MSELoss

from .models import (
    MLP_generator_net,
    MLP_preact_generator_net,
    MLP_preact_recognition_net,
    MLP_recognition_net,
)


class VAE_gauss(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim, pred_sig=True):
        super(VAE_gauss, self).__init__()

        self.encoder = MLP_preact_recognition_net(input_dim, width, depth, latent_dim)
        if pred_sig:
            self.decoder = MLP_preact_generator_net(
                2 * input_dim, width, depth, latent_dim
            )
            self.rec_loglike = GaussianLoglike(min_sigma=1e-2)
        else:
            self.decoder = MLP_preact_generator_net(input_dim, width, depth, latent_dim)
            self.m_rec_loglike = MSELoss(reduction="none")
        self.pred_sig = pred_sig

    def encode(self, x):
        approx_post_params = self.encoder(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def decode(self, z_sample):
        rec_params = self.decoder(z_sample)
        return rec_params

    def vlb(self, prior, approx_post, x, rec_params):
        if self.pred_sig:
            rec = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        else:
            rec = -self.m_rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        return rec - kl

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            if self.pred_sig:
                rec_loglike = (
                    self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
                )
            else:
                rec_loglike = (
                    -self.m_rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
                )

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(x.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = approx_post.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(x.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loglike + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - np.log(K)


class VAE_gauss_net(BaseNet):
    def __init__(
        self, input_dim, width, depth, latent_dim, pred_sig=True, lr=1e-3, cuda=True
    ):
        super(VAE_gauss_net, self).__init__()
        cprint("y", "VAE_gauss_net")

        self.cuda = cuda

        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr
        self.pred_sig = pred_sig

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        if self.cuda:
            self.prior = self.prior = Normal(
                loc=torch.zeros(latent_dim).cuda(), scale=torch.ones(latent_dim).cuda()
            )
        else:
            self.prior = Normal(
                loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim)
            )
        self.vlb_scale = (
            1 / input_dim
        )  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAE_gauss(
            self.input_dim, self.width, self.depth, self.latent_dim, self.pred_sig
        )
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        print("    Total params: %.2fM" % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr)

    def fit(self, x):
        self.set_mode_train(train=True)

        (x,) = to_variable(var=(x,), cuda=self.cuda)
        self.optimizer.zero_grad()

        approx_post = self.model.encode(x)
        z_sample = approx_post.rsample()
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)
        loss = (-vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), rec_params

    def eval(self, x, sample=False):
        self.set_mode_train(train=False)

        (x,) = to_variable(var=(x,), cuda=self.cuda)
        approx_post = self.model.encode(x)
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)

        return vlb.mean().item(), rec_params

    def eval_iw(self, x, k=50):
        self.set_mode_train(train=False)
        (x,) = to_variable(var=(x,), cuda=self.cuda)

        approx_post = self.model.recognition_encode(x)

        iw_lb = self.model.iwlb(self.prior, approx_post, x, k)
        return iw_lb.mean().item()

    def recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            (x,) = to_variable(var=(x,), volatile=True, cuda=self.cuda)
        approx_post = self.model.encode(x)
        return approx_post

    def regenerate(self, z, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            (z,) = to_variable(var=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)
        if self.pred_sig:
            return normal_parse_params(out, 1e-2)
        else:
            return out
