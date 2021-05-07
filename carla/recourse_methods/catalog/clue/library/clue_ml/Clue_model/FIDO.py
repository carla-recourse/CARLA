from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from src.probability import decompose_entropy_cat, decompose_std_gauss
from src.utils import BaseNet, MNIST_mean_std_norm, to_variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# TODO: Might not be dividing loss by the number of samples, effectively adding unecesary variance


def gumbel_sigmoid(prob_map, temperature, eps=1e-20):
    U = prob_map.new(prob_map.shape).uniform_(0, 1)
    sigmoid_in = (
        torch.log(prob_map + eps)
        - torch.log(1 - prob_map + eps)
        + torch.log(U + eps)
        - torch.log(1 - U + eps)
    )
    y = torch.sigmoid(sigmoid_in / temperature)
    y_hard = torch.round(y)
    return (y_hard - y).detach() + y


class bern_mask(nn.Module):
    def __init__(self, shape, init_p=0.5, temp=0.1):
        super(bern_mask, self).__init__()

        self.mask_probs = nn.Parameter(torch.ones(shape) * init_p)
        self.temp = temp

    def forward(self, x):
        hard_mask = gumbel_sigmoid(self.mask_probs, self.temp)
        return x * hard_mask, (1 - hard_mask)


class mask_explainer(BaseNet):
    def __init__(
        self,
        shape,
        mask_L1_weight,
        aleatoric_coeff,
        epistemic_coeff,
        mask_samples=1,
        lr=0.05,
        decay_period=5,
        gamma=0.8,
        cuda=True,
    ):
        super(mask_explainer, self).__init__()

        self.mask_L1_weight = mask_L1_weight
        self.aleatoric_coeff = aleatoric_coeff
        self.epistemic_coeff = epistemic_coeff
        self.mask_samples = mask_samples

        self.model = bern_mask(shape, init_p=0.5, temp=0.1)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=decay_period, gamma=gamma)

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def fit_cat(self, x, BNN, VAEAC, flatten_ims=True, test_dims=None, plot=False):
        # note that x will need to be the same shape as specified at class initialisation
        (x,) = to_variable(var=(x,), cuda=self.cuda)
        self.set_mode_train(train=True)
        BNN.set_mode_train(train=False)
        VAEAC.set_mode_train(train=False)

        self.optimizer.zero_grad()
        loss_cum = 0
        aleatoric_cum = 0
        epistemic_cum = 0

        for it in range(self.mask_samples):

            masked_x, mask = self.model(x)

            if flatten_ims:
                flat_x = x.view(masked_x.shape[0], -1)
                masked_x = masked_x.view(masked_x.shape[0], -1)
                mask = mask.view(mask.shape[0], -1)
            else:
                flat_x = x
            if test_dims is not None:
                #                 x = torch.cat([x, x.new_zeros(x.shape[0], test_dims)], dim=1)
                masked_x = torch.cat(
                    [masked_x, masked_x.new_zeros(masked_x.shape[0], test_dims)], dim=1
                )
                mask = torch.cat([mask, mask.new_ones(mask.shape[0], test_dims)], dim=1)

            # We dont want gradients from this
            inpainted = VAEAC.inpaint(
                masked_x.data, mask.data, Nsample=1, z_mean=True
            ).data.squeeze(0)
            if test_dims is not None:
                inpainted = inpainted[:, :-test_dims]
                mask = mask[:, :-test_dims]

            to_BNN = inpainted * mask + flat_x * (1 - mask)
            to_BNN = MNIST_mean_std_norm(to_BNN)

            probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(
                probs
            )

            # We mean across batch
            aleatoric_cum += aleatoric_entropy.mean().item()
            epistemic_cum += epistemic_entropy.mean().item()
            # we should average over MC samples but sum over batch and features
            loss = (
                self.aleatoric_coeff * aleatoric_entropy
                + self.epistemic_coeff * epistemic_entropy
                + self.mask_L1_weight * mask.sum(dim=1)
            ).sum(dim=0) / self.mask_samples
            loss.backward()  # Gradient accumulation
            loss_cum += loss.item() / aleatoric_entropy.shape[0]

        # we average here to be invariant to number of samples
        aleatoric_cum = aleatoric_cum / self.mask_samples
        epistemic_cum = epistemic_cum / self.mask_samples

        self.optimizer.step()
        self.model.mask_probs.data = torch.clamp(self.model.mask_probs, min=0, max=1)
        self.scheduler.step()

        return loss_cum, aleatoric_cum, epistemic_cum

    def fit_gauss(self, x, BNN, VAEAC, flatten_ims=True, test_dims=None, plot=False):
        (x,) = to_variable(var=(x,), cuda=self.cuda)
        self.set_mode_train(train=True)
        BNN.set_mode_train(train=False)
        VAEAC.set_mode_train(train=False)

        self.optimizer.zero_grad()
        loss_cum = 0
        aleatoric_cum = 0
        epistemic_cum = 0

        for it in range(self.mask_samples):

            masked_x, mask = self.model(x)

            if flatten_ims:
                flat_x = x.view(masked_x.shape[0], -1)
                masked_x = masked_x.view(masked_x.shape[0], -1)
                mask = mask.view(mask.shape[0], -1)
            else:
                flat_x = x
            if test_dims is not None:
                #                 x = torch.cat([x, x.new_zeros(x.shape[0], test_dims)], dim=1)
                masked_x = torch.cat(
                    [masked_x, masked_x.new_zeros(masked_x.shape[0], test_dims)], dim=1
                )
                mask = torch.cat([mask, mask.new_ones(mask.shape[0], test_dims)], dim=1)

            # We dont want gradients from this
            # Switched to non gauss output
            inpainted = VAEAC.inpaint(
                masked_x.data, mask.data, Nsample=1, z_mean=True
            ).data.squeeze(0)
            if test_dims is not None:
                inpainted = inpainted[:, :-test_dims]
                mask = mask[:, :-test_dims]

            to_BNN = inpainted * mask + flat_x * (1 - mask)

            mu, std = BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
            total_std, aleatoric_std, epistemic_std = decompose_std_gauss(mu, std)

            # we average here to be invariant to batch size
            aleatoric_cum += aleatoric_std.mean().item()
            epistemic_cum += epistemic_std.mean().item()
            loss = (
                self.aleatoric_coeff * aleatoric_std
                + self.epistemic_coeff * epistemic_std
                + self.mask_L1_weight * mask.sum(dim=1)
            ).sum(dim=0) / self.mask_samples
            loss.backward()  # Gradient accumulation
            loss_cum += loss.item() / aleatoric_std.shape[0]

        # we average here to be invariant to number of samples
        aleatoric_cum = aleatoric_cum / self.mask_samples
        epistemic_cum = epistemic_cum / self.mask_samples

        #         loss_cum.backward()
        self.optimizer.step()
        self.model.mask_probs.data = torch.clamp(self.model.mask_probs, min=0, max=1)
        self.scheduler.step()

        return loss_cum, aleatoric_cum, epistemic_cum

    def get_mask(self):
        self.set_mode_train(train=False)
        """Note that this returns 1s for input features which are masked"""
        return 1 - self.model.mask_probs.data.round()

    def get_mask_probs(self):
        self.set_mode_train(train=False)
        """Note that this returns 1s for input features which are masked"""
        return 1 - self.model.mask_probs.data

    def mask_input(self, x):
        self.set_mode_train(train=False)
        (x,) = to_variable(var=(x,), cuda=self.cuda)
        self.set_mode_train(train=False)
        return x * self.model.mask_probs.data.round()

    def mask_inpaint(self, x, VAEAC, flatten_ims=True, test_dims=None, cat=False):
        (x,) = to_variable(var=(x,), cuda=self.cuda)
        self.set_mode_train(train=False)
        VAEAC.set_mode_train(train=False)

        masked_x = x * self.model.mask_probs.data.round().data
        mask = 1 - self.model.mask_probs.data.round().data

        if flatten_ims:
            flat_x = x.view(masked_x.shape[0], -1)
            masked_x = masked_x.view(masked_x.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)
        else:
            flat_x = x

        if test_dims is not None:
            masked_x = torch.cat(
                [masked_x, masked_x.new_zeros(masked_x.shape[0], test_dims)], dim=1
            )
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], test_dims)], dim=1)

        # We dont want gradients from this
        if cat:
            inpainted = VAEAC.inpaint(
                masked_x.data, mask.data, Nsample=1, z_mean=True
            ).data.squeeze(0)
        else:
            inpainted = VAEAC.inpaint(
                masked_x.data, mask.data, Nsample=1, z_mean=True
            ).data.squeeze(0)
        if test_dims is not None:
            inpainted = inpainted[:, :-test_dims]
            mask = mask[:, :-test_dims]

        out = inpainted * mask + flat_x * (1 - mask)

        return out.data, mask.data

    @staticmethod
    def train_mask(
        x,
        BNN,
        VAEAC,
        aleatoric_coeff,
        epistemic_coeff,
        L1w=1,
        N_epochs=30,
        mask_samples=20,
        mask_samples2=10,
        cat=True,
        flatten_ims=True,
        test_dims=None,
    ):

        torch.cuda.empty_cache()
        x_pixels = x.view(x.shape[0], -1).shape[1]

        explainer = mask_explainer(
            shape=x.shape,
            mask_L1_weight=L1w / x_pixels,
            aleatoric_coeff=aleatoric_coeff,
            epistemic_coeff=epistemic_coeff,
            mask_samples=mask_samples,
            lr=0.05,
            decay_period=5,
            gamma=0.8,
            cuda=True,
        )

        loss_vec = []
        aleatoric_vec = []
        epistemic_vec = []
        for i in range(N_epochs):
            if (
                i > 10
            ):  # We train with more samples at the beginning as there is a lot more variability
                explainer.mask_samples = mask_samples2
            if cat:
                loss, aleatoric_ent, epistemic_ent = explainer.fit_cat(
                    x,
                    BNN,
                    VAEAC,
                    flatten_ims=flatten_ims,
                    test_dims=test_dims,
                    plot=False,
                )
            else:
                loss, aleatoric_ent, epistemic_ent = explainer.fit_gauss(
                    x,
                    BNN,
                    VAEAC,
                    flatten_ims=flatten_ims,
                    test_dims=test_dims,
                    plot=False,
                )

            loss_vec.append(loss)
            aleatoric_vec.append(aleatoric_ent)
            epistemic_vec.append(epistemic_ent)
            print(
                "it: %d, loss: %3.3f, aleatoric: %3.3f, epistemic: %3.3f"
                % (i, loss, aleatoric_ent, epistemic_ent)
            )

        loss_vec = np.array(loss_vec)
        aleatoric_vec = np.array(aleatoric_vec)
        epistemic_vec = np.array(epistemic_vec)

        return explainer, loss_vec, aleatoric_vec, epistemic_vec
