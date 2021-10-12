from __future__ import division

import numpy as np
import torch.backends.cudnn as cudnn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from carla import log
from carla.recourse_methods.catalog.clue.library.clue_ml.src.gauss_cat import *
from carla.recourse_methods.catalog.clue.library.clue_ml.src.probability import (
    normal_parse_params,
)
from carla.recourse_methods.catalog.clue.library.clue_ml.src.radam import RAdam
from carla.recourse_methods.catalog.clue.library.clue_ml.src.utils import (
    BaseNet,
    to_variable,
)

from .models import MLP_preact_generator_net, MLP_preact_recognition_net

# TODO: implement for std changeable gaussian instead of rms


class VAE_gauss_cat(nn.Module):
    def __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=False):
        super(VAE_gauss_cat, self).__init__()

        input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in input_dim_vec:
            input_dim += e

        self.encoder = MLP_preact_recognition_net(input_dim, width, depth, latent_dim)
        if pred_sig:
            raise NotImplementedError()
        else:
            self.decoder = MLP_preact_generator_net(input_dim, width, depth, latent_dim)
            self.rec_loglike = rms_cat_loglike(self.input_dim_vec, reduction="none")
        self.pred_sig = pred_sig

    def encode(self, x):
        """Works with flattened representATION"""
        approx_post_params = self.encoder(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def decode(self, z_sample):
        """Works with flattened representATION"""
        rec_params = self.decoder(z_sample)
        return rec_params

    def vlb(self, prior, approx_post, x, rec_params):
        """Works with flattened representATION"""
        if self.pred_sig:
            pass
        else:
            rec = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        return rec - kl

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            if self.pred_sig:
                pass
            else:
                rec_loglike = (
                    self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
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


class VAE_gauss_cat_net(BaseNet):
    def __init__(
        self,
        input_dim_vec,
        width,
        depth,
        latent_dim,
        pred_sig=False,
        lr=1e-3,
        cuda=True,
        flatten=True,
    ):
        super(VAE_gauss_cat_net, self).__init__()
        log.info("VAE_gauss_net")

        self.cuda = cuda
        self.input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in self.input_dim_vec:
            self.input_dim += e
        self.flatten = flatten
        if not self.flatten:
            pass

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
        self.vlb_scale = 1 / len(
            self.input_dim_vec
        )  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAE_gauss_cat(
            self.input_dim_vec, self.width, self.depth, self.latent_dim, self.pred_sig
        )
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        log.info("Total params: %.2fM" % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr)

    def fit(self, x):
        self.set_mode_train(train=True)

        if self.flatten:
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x, x_flat = to_variable(var=(x, x_flat), cuda=self.cuda)
        self.optimizer.zero_grad()

        approx_post = self.model.encode(x_flat)
        z_sample = approx_post.rsample()
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)
        loss = (-vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), rec_params

    def eval(self, x, sample=False):
        self.set_mode_train(train=False)

        if self.flatten:
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x, x_flat = to_variable(var=(x, x_flat), cuda=self.cuda)
        approx_post = self.model.encode(x_flat)
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)

        return vlb.mean().item(), rec_params

    def eval_iw(self, x, k=50):
        self.set_mode_train(train=False)
        if self.flatten:
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x, x_flat = to_variable(var=(x, x_flat), cuda=self.cuda)

        approx_post = self.model.recognition_encode(x)

        iw_lb = self.model.iwlb(self.prior, approx_post, x, k)
        return iw_lb.mean().item()

    def recongnition(self, x, grad=False, flatten=None):
        if flatten is None:
            flatten = self.flatten
        if flatten and grad:
            raise Exception("flatten and grad options are not compatible")
        self.set_mode_train(train=False)
        if flatten:
            x = gauss_cat_to_flat(x, self.input_dim_vec)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            (x,) = to_variable(var=(x,), volatile=True, cuda=self.cuda)
        approx_post = self.model.encode(x)
        return approx_post

    def regenerate(self, z, grad=False, unflatten=False):
        if unflatten and grad:
            raise Exception("flatten and grad options are not compatible")
        self.set_mode_train(train=False)
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            (z,) = to_variable(var=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)

        if unflatten:
            out = flat_to_gauss_cat(out, self.input_dim_vec)
        else:
            out = selective_softmax(out, self.input_dim_vec, grad=grad)

        if self.pred_sig:
            raise Exception("Not implemented")
        else:
            return out
