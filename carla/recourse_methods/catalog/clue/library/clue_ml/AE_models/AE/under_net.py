from __future__ import division

import torch
import torch.backends.cudnn as cudnn
from src.probability import normal_parse_params
from src.radam import RAdam
from src.utils import BaseNet, cprint, to_variable
from torch.distributions.normal import Normal

from .fc_gauss import VAE_gauss


class under_VAE(
    BaseNet
):  # TODO: because of behaviour of class, should be called under_VAE_net
    def __init__(self, base_VAE, width, depth, latent_dim, lr=1e-3, cuda=True):
        super(under_VAE, self).__init__()
        cprint("y", "VAE_gauss_net")

        self.base_VAE = base_VAE
        self.pred_sig = False

        self.cuda = cuda

        self.input_dim = self.base_VAE.latent_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr

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
            1 / self.input_dim
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

        z_sample = self.base_VAE.encode(x).sample()
        approx_post = self.model.encode(z_sample)
        u_sample = approx_post.rsample()
        rec_params = self.model.decode(u_sample)

        vlb = self.model.vlb(self.prior, approx_post, z_sample, rec_params)
        loss = (-vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), rec_params

    def eval(self, x):
        self.set_mode_train(train=False)

        (x,) = to_variable(var=(x,), cuda=self.cuda)
        z_sample = self.base_VAE.encode(x).sample()
        approx_post = self.model.encode(z_sample)
        u_sample = approx_post.rsample()
        rec_params = self.model.decode(u_sample)

        vlb = self.model.vlb(self.prior, approx_post, z_sample, rec_params)

        return vlb.mean().item(), rec_params

    def eval_iw(self, x, k=50):
        self.set_mode_train(train=False)
        (x,) = to_variable(var=(x,), cuda=self.cuda)
        z_sample = self.base_VAE.encode(x).sample()
        approx_post = self.model.recognition_encode(z_sample)

        iw_lb = self.model.iwlb(self.prior, approx_post, z_sample, k)
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
        return out.data

    def u_recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            (x,) = to_variable(var=(x,), volatile=True, cuda=self.cuda)

        z = self.base_VAE.encode(x).loc
        approx_post = self.model.encode(z)
        return approx_post

    def u_regenerate(self, u, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not u.requires_grad:
                u.requires_grad = True
        else:
            (u,) = to_variable(var=(u,), volatile=True, cuda=self.cuda)

        z = self.model.decode(u)
        out = self.base_VAE.decode(z)
        if self.base_VAE.pred_sig:
            return normal_parse_params(out, 1e-2)
        else:
            return out.data
