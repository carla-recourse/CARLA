from __future__ import division

import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import choice
from src.utils import Ln_distance

# TODO: percentile gap creation by feature


class dataset_editor:
    def __init__(self, full_trainset, full_testset):
        self.full_trainset = copy.deepcopy(full_trainset)
        self.full_testset = copy.deepcopy(full_testset)

    def random_train_subset(self, nkeep):
        all_idxs = np.arange(len(self.full_trainset))
        chosen_idxs = choice(
            np.arange(len(self.full_trainset)), size=(nkeep), replace=False, p=None
        )
        held_out_idxs = np.delete(all_idxs, chosen_idxs)

        small_train = copy.deepcopy(self.full_trainset)
        held_out_train = copy.deepcopy(self.full_trainset)

        small_train.data = small_train.data[chosen_idxs]
        small_train.targets = small_train.targets[chosen_idxs]

        held_out_train.data = held_out_train.data[held_out_idxs]
        held_out_train.targets = held_out_train.targets[held_out_idxs]

        return small_train, held_out_train

    def keep_by_target(self, target_vec, train=True):

        if train:
            # trainset
            keep_idx = None
            for t_idx, t in enumerate(target_vec):

                if keep_idx is None:
                    keep_idx = self.full_trainset.targets == t
                else:
                    keep_idx = keep_idx | (self.full_trainset.targets == t)

            self.full_trainset.targets = self.full_trainset.targets[keep_idx]
            self.full_trainset.data = self.full_trainset.data[keep_idx]

            for t_idx, t in enumerate(target_vec):

                idx0 = self.full_trainset.targets == t
                self.full_trainset.targets[idx0] = t_idx
        else:
            # testset
            keep_idx = None
            for t_idx, t in enumerate(target_vec):

                if keep_idx is None:
                    keep_idx = self.full_testset.targets == t
                else:
                    keep_idx = keep_idx | (self.full_testset.targets == t)

            self.full_testset.targets = self.full_testset.targets[keep_idx]
            self.full_testset.data = self.full_testset.data[keep_idx]

            for t_idx, t in enumerate(target_vec):

                idx0 = self.full_testset.targets == t
                self.full_testset.targets[idx0] = t_idx

    def split_by_value(self, nvars, values, train=True):
        if train:
            dset = copy.deepcopy(self.full_trainset)
        else:
            dset = copy.deepcopy(self.full_testset)
            # TODO: implement this function
        pass

    def get_by_index(self, idxs, train=True):
        if train:
            dset = copy.deepcopy(self.full_trainset)
        else:
            dset = copy.deepcopy(self.full_testset)
        dset.data = dset.data[idxs]
        dset.targets = dset.targets[idxs]
        return dset

    def remove_by_index(self, idxs, train=True):
        if train:
            dset = self.full_trainset
        else:
            dset = self.full_testset

        all_idxs = np.arange(len(dset))
        keep_idxs = np.delete(all_idxs, idxs)

        if train:
            self.full_trainset.data = self.full_trainset.data[keep_idxs]
            self.full_trainset.targets = self.full_trainset.targets[keep_idxs]
        else:
            self.full_testset.data = self.full_testset.data[keep_idxs]
            self.full_testset.targets = self.full_testset.targets[keep_idxs]

        return None

    def notebook_display(
        self, labels=None, nview=0, Nrows=4, Ncols=5, train=True, dpi=200, hspace=5
    ):
        if train:
            data = self.full_trainset.data
        else:
            data = self.full_testset.data

        dataset_editor.notebook_display_s(
            data,
            labels=labels,
            nview=nview,
            Nrows=Nrows,
            Ncols=Ncols,
            dpi=dpi,
            hspace=hspace,
        )

    @staticmethod
    def notebook_display_s(
        data, labels=None, nview=0, Nrows=4, Ncols=5, dpi=200, hspace=5
    ):

        if nview == 0:
            nview = data.shape[0]

        im_idx = 0

        while 1:

            fig = plt.figure(figsize=(Ncols, Nrows), dpi=dpi)
            im = []
            for i in range(Ncols * Nrows):
                if im_idx + i == nview:
                    break
                dd = data[im_idx + i]
                plt.subplots_adjust(hspace=hspace)
                ax1 = fig.add_subplot(Nrows, Ncols, i + 1)
                im.append(ax1.imshow(dd.numpy(), interpolation="nearest"))
                ax1.set_title(str(im_idx + i))
                if labels is not None:
                    ax1.set_xlabel(str(labels[im_idx + i]))
                # ax1.axis('off')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
            fig.tight_layout()
            plt.show()
            im_idx += Ncols * Nrows
            if im_idx >= nview:
                break

            if sys.version_info > (3, 0):
                input("Press Enter to continue...")
            else:
                raw_input("Press Enter to continue...")

    def get_KNN(self, samples, K, train=True, dist=None, mean=False):
        """If dist is None, pairwise L2 will be used.
        The dimension 0 of samples should be the number of samples
        If the mean flag is set, the returned points will be close to all input samples, else a different k will
            be given per sample
        """
        if train:
            data = self.full_trainset.data
            tgets = self.full_trainset.targets
        else:
            data = self.full_testset.data
            tgets = self.full_testset.targets

        # data (1, Npoints, dims)
        # samples (Npoints, 1, dims)
        if dist is None:
            dist = Ln_distance(n=2, dim=(2))

        samples = samples.view(samples.shape[0], -1).type(torch.FloatTensor)
        data = data.view(data.shape[0], -1).type(torch.FloatTensor)

        pairwise = dist(samples.unsqueeze(dim=1), data.unsqueeze(dim=0)).type(
            torch.FloatTensor
        )

        if mean:
            pairwise = pairwise.mean(dim=0, keepdim=False)
            sorted_idxs = torch.argsort(pairwise, dim=0, descending=False)
            idxs = sorted_idxs[:K]
            return idxs, data[idxs], tgets[idxs]
        else:
            sorted_idx_stack = []
            output = []
            tgets_out = []
            # pairwise (samples, data)
            for i in range(pairwise.shape[0]):
                sorted_idxs = torch.argsort(pairwise[i, :], descending=False)[:K]

                sorted_idx_stack.append(sorted_idxs.unsqueeze(0))
                output.append(data[sorted_idxs].unsqueeze(0))
                tgets_out.append(tgets[sorted_idxs].unsqueeze(0))

            output = torch.cat(output, dim=0)
            sorted_idx_stack = torch.cat(sorted_idx_stack, dim=0)
            tgets_out = torch.cat(tgets_out, dim=0)

            return sorted_idx_stack, output, tgets_out
