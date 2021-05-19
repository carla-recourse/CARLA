from __future__ import division, print_function

import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    import cPickle as pickle  # type: ignore
except:
    import pickle  # type: ignore


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)


import torch.nn as nn

suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = "%.2f" % nbytes
    return "%s%s" % (f, suffixes[i])


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return (
            nb_samples + (-nb_samples % batch_size)
        ) / batch_size  # roundup division
    else:
        return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size : (i + 1) * batch_size]


def to_variable(var=(), cuda=False, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)
        if not v.is_cuda and cuda:
            v = v.cuda()
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        out.append(v)
    return out


def cprint(color, text, **kwargs):
    if color[0] == "*":
        pre_code = "1;"
        color = color[1:]
    else:
        pre_code = ""
    code = {
        "a": "30",
        "r": "31",
        "g": "32",
        "y": "33",
        "b": "34",
        "p": "35",
        "c": "36",
        "w": "37",
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


import numpy as np
import torch.utils.data as data


class Datafeed(data.Dataset):
    def __init__(self, x_train, y_train=None, transform=None):
        self.data = x_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[index]
        else:
            return img

    def __len__(self):
        return len(self.data)


# ----------------------------------------------------------------------------------------------------------------------
class BaseNet(object):
    def __init__(self):
        cprint("c", "\nNet:")

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print("learning rate: %f  (%d)\n" % (self.lr, epoch))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr

    def save(self, filename):
        cprint("c", "Writting %s\n" % filename)
        torch.save(
            {
                "epoch": self.epoch,
                "lr": self.lr,
                "model": self.model,
                "optimizer": self.optimizer,
            },
            filename,
        )

    def load(self, filename):
        cprint("c", "Reading %s\n" % filename)
        state_dict = torch.load(filename)  # added map_location
        self.epoch = state_dict["epoch"]
        self.lr = state_dict["lr"]
        self.model = state_dict["model"]
        self.optimizer = state_dict["optimizer"]
        print("  restoring epoch: %d, lr: %f" % (self.epoch, self.lr))
        return self.epoch


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    #     perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def torch_onehot(y, Nclass):
    if y.is_cuda:
        y = y.type(torch.cuda.LongTensor)
    else:
        y = y.type(torch.LongTensor)
    y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
    # In your for loop
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


# def save_object(obj, filename):
#    with open(filename, 'wb') as output:  # Overwrites any existing file.
#        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# This one also led to the magic number error
# def save_object(obj, filename):
#    with open(filename, 'wb') as output:  # Overwrites any existing file.
#        torch.save(obj, output)

# Changed to this: https://discuss.pytorch.org/t/runtime-error-when-loading-pytorch-model-from-pkl/38330
def save_object(obj, filename):
    torch.save(obj, filename)


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


import contextlib

_map_location = [None]


@contextlib.contextmanager
def map_location(location):
    _map_location.append(location)
    yield
    _map_location.pop()


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location=_map_location[-1])


# def load_object(filename):
#    with torch.storage.map_location('cpu'):
#        return pickle.load(open(filename, 'rb'), encoding='bytes')


def load_object(filename):
    with open(filename, "rb") as input:
        ##try:
        print(input)
        return torch.load(input, encoding="bytes")

        """
        except Exception as e:
            try:
                print(e)
                return pickle.load(input, encoding="latin1")
            except Exception as a:
                print(a)
                return pickle.load(input, encoding='bytes')
        """


def array_to_bin_np(array, ncats):
    array = np.array(array)
    bin_vec = np.zeros(ncats)
    bin_vec[array] = 1
    return bin_vec.astype(bool)


def MNIST_mean_std_norm(x):
    mean = 0.1307
    std = 0.3081
    x = x - mean
    x = x / std
    return x


def complete_logit_norm_vec(vec):
    last_term = 1 - vec.sum(dim=1, keepdim=True)
    cvec = torch.cat((vec, last_term), dim=1)
    return cvec


class Ln_distance(nn.Module):
    """If dims is None Compute across all dimensions except first"""

    def __init__(self, n, dim=None):
        super(Ln_distance, self).__init__()
        self.n = n
        self.dim = dim

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return torch.abs(d).pow(self.n).sum(dim=self.dim).pow(1.0 / float(self.n))


def smooth_median(X, dim=0):
    """Just gets numpy behaviour instead of torch default
    dim is dimension to be reduced, across which median is taken"""
    yt = X.clone()
    ymax = yt.max(dim=dim, keepdim=True)[
        0
    ]  # maybe this is wrong  and dont need keepdim
    yt_exp = torch.cat((yt, ymax), dim=dim)
    smooth_median = (yt_exp.median(dim=dim)[0] + yt.median(dim=dim)[0]) / 2.0
    return smooth_median


class l1_MAD(nn.Module):
    """Intuition behind this metric -> allows variability only where the dataset has variability
    Otherwise it penalises discrepancy heavily. Might not make much sense if data is already normalised to
    unit std. Might also not make sense if we want to detect outlier values in specific features."""

    def __init__(self, trainset_data, median_dim=0, dim=None):
        """Median dim are those across whcih to normalise (not features)
        dim is dimension to sum (features)"""
        super(l1_MAD, self).__init__()
        self.dim = dim
        feature_median = smooth_median(trainset_data, dim=median_dim).unsqueeze(
            dim=median_dim
        )
        self.MAD = smooth_median(
            (trainset_data - feature_median).abs(), dim=median_dim
        ).unsqueeze(dim=median_dim)
        self.MAD = self.MAD.clamp(min=1e-4)

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return (torch.abs(d) / self.MAD).sum(dim=self.dim)
