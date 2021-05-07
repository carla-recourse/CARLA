from __future__ import division

import numpy as np
import torch
from src.dataset_editor import dataset_editor
from src.utils import Ln_distance
from torchvision import datasets, transforms


def get_mod_479_MNIST(root="../data", transforms=None):

    trainset = datasets.MNIST(
        root=root, train=True, download=True, transform=transforms
    )
    valset = datasets.MNIST(root=root, train=False, download=True, transform=transforms)

    deditor = dataset_editor(trainset, valset)

    deditor.keep_by_target([4, 7, 9])
    deditor.keep_by_target([4, 7, 9], train=False)
    print("left dset size %d" % len(deditor.full_trainset))
    print("left testset size %d" % len(deditor.full_testset))

    hold_out4 = [
        16,
        46,
        135,
        183,
        332,
        351,
        385,
        505,
        515,
        555,
        587,
        603,
        705,
        745,
        766,
        789,
        800,
        876,
        910,
        1057,
        1206,
        1217,
        1406,
        1502,
        1637,
        1646,
        1631,
        1657,
        1664,
        1666,
        1667,
        1672,
        1679,
        1790,
        1922,
        1958,
        2101,
        2107,
        2163,
        2291,
        2333,
    ]

    print("N prototypes 4", len(hold_out4))

    hold_out7 = [
        10,
        45,
        100,
        127,
        153,
        162,
        230,
        284,
        305,
        306,
        307,
        350,
        439,
        630,
        631,
        652,
        668,
        676,
        720,
        721,
        757,
        800,
        834,
        841,
        846,
        857,
        931,
        936,
        1008,
        1024,
        1034,
        1106,
        1122,
        1167,
        1189,
        1212,
        1215,
        1220,
        1245,
        1273,
        1295,
        1352,
        1412,
        1433,
        1459,
        1788,
        1505,
    ]

    print("N prototypes 7", len(hold_out7))

    samples4 = deditor.full_trainset.data[hold_out4]
    samples7 = deditor.full_trainset.data[hold_out7]

    dist4 = Ln_distance(n=1, dim=(2))
    dist7 = Ln_distance(n=2, dim=(2))

    KNN_idxs_4, KNN_samples, KNN_targets = deditor.get_KNN(
        samples4, K=15, train=True, dist=dist4, mean=False
    )
    KNN_idxs_4 = KNN_idxs_4.view(-1)

    KNN_idxs_7, KNN_samples, KNN_targets = deditor.get_KNN(
        samples7, K=12, train=True, dist=dist7, mean=False
    )
    KNN_idxs_7 = KNN_idxs_7.view(-1)

    correct_4_idxs = (deditor.full_trainset.targets[KNN_idxs_4] == 0).type(
        torch.ByteTensor
    )
    KNN_idxs_4 = KNN_idxs_4[correct_4_idxs]

    correct_7_idxs = (deditor.full_trainset.targets[KNN_idxs_7] == 1).type(
        torch.ByteTensor
    )
    KNN_idxs_7 = KNN_idxs_7[correct_7_idxs]

    delete_idxs = torch.cat([KNN_idxs_7, KNN_idxs_4], dim=0)

    # print('full idx to remove', delete_idxs.shape)

    print("prev Ntrain, ", len(deditor.full_trainset))
    deditor.remove_by_index(delete_idxs, train=True)
    print("post Ntrain, ", len(deditor.full_trainset))
    return deditor.full_trainset, deditor.full_testset
