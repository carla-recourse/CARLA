from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.probability import decompose_entropy_cat, decompose_std_gauss
from src.utils import MNIST_mean_std_norm, generate_ind_batch
from torchvision.utils import make_grid, save_image


def latent_map_2d_gauss(BNN, VAE, vae_sig=True, steps=300, extent=4, batch_size=2048):
    """Returns numpy matrices for aleatoric and epistemic entropy sweeps of latent space around
    specified coordinates. Requieres Gaussian output VAE and BNN."""
    dim_range = np.linspace(-extent, extent, steps)
    dimx, dimy = np.meshgrid(dim_range, dim_range)
    dim_mtx = np.concatenate((np.expand_dims(dimx, 2), np.expand_dims(dimy, 2)), axis=2)

    z = torch.from_numpy(dim_mtx).type(torch.FloatTensor).cuda()
    z = z.view(steps ** 2, 2)
    z_mat = z.data.cpu().numpy()

    iterator = generate_ind_batch(z.shape[0], batch_size, random=False, roundup=True)

    entropy_vec = []
    aleatoric_vec = []
    epistemic_vec = []

    for idx_set in iterator:

        if vae_sig:
            x = VAE.regenerate(z[idx_set, :], grad=False).loc.cpu()
        else:
            x = VAE.regenerate(z[idx_set, :], grad=False).cpu()
        mu_vec, std_vec = BNN.sample_predict(x, 0, False)

        total_entropy, aleatoric_entropy, epistemic_entropy = decompose_std_gauss(
            mu_vec, std_vec
        )

        entropy_vec.append(total_entropy.data)
        aleatoric_vec.append(aleatoric_entropy.data)
        epistemic_vec.append(epistemic_entropy.data)

    entropy_vec = torch.cat(entropy_vec).view(steps, steps).cpu().numpy()
    aleatoric_vec = torch.cat(aleatoric_vec).view(steps, steps).cpu().numpy()
    epistemic_vec = torch.cat(epistemic_vec).view(steps, steps).cpu().numpy()

    return z_mat, entropy_vec, aleatoric_vec, epistemic_vec


def latent_project_gauss(BNN, VAE, dset, batch_size=1024, cuda=True, prob_BNN=False):
    if cuda:
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
        )
    else:
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )
    z_train = []
    y_train = []
    x_train = []
    tr_aleatoric_vec = []
    tr_epistemic_vec = []

    for j, (x, y_l) in enumerate(loader):
        zz = VAE.recongnition(x).loc.data.cpu().numpy()
        # Note that naming is wrong and this is actually std instead of entropy
        print(zz)
        print(x)
        if prob_BNN:
            mu_vec, std_vec = BNN.sample_predict(x, 0, False)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_std_gauss(
                mu_vec, std_vec
            )
        else:
            mu, std = BNN.predict(x, grad=False)
            total_entropy = std
            aleatoric_entropy = std
            epistemic_entropy = std * 0

        tr_epistemic_vec.append(epistemic_entropy.data)
        tr_aleatoric_vec.append(aleatoric_entropy.data)

        z_train.append(zz)
        y_train.append(y_l.numpy())
        x_train.append(x.numpy())

    tr_aleatoric_vec = torch.cat(tr_aleatoric_vec).cpu().numpy()
    tr_epistemic_vec = torch.cat(tr_epistemic_vec).cpu().numpy()
    z_train = np.concatenate(z_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train


def latent_project_cat(BNN, VAE, dset, batch_size=1024, cuda=True, prob_BNN=True):
    if cuda:
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
        )
    else:
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )
    z_train = []
    y_train = []
    x_train = []
    tr_aleatoric_vec = []
    tr_epistemic_vec = []

    for j, (x, y_l) in enumerate(loader):
        zz = VAE.recongnition(x).loc.data.cpu().numpy()

        # print(x.shape)
        if prob_BNN:
            probs = BNN.sample_predict(x, 0, False)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(
                probs
            )
        else:
            probs = BNN.predict(x, grad=False)
            total_entropy = -(probs * torch.log(probs + 1e-10)).sum(
                dim=1, keepdim=False
            )
            aleatoric_entropy = total_entropy
            epistemic_entropy = total_entropy * 0

        tr_epistemic_vec.append(epistemic_entropy.data)
        tr_aleatoric_vec.append(aleatoric_entropy.data)

        z_train.append(zz)
        y_train.append(y_l.numpy())
        x_train.append(x.numpy())

    tr_aleatoric_vec = torch.cat(tr_aleatoric_vec).cpu().numpy()
    tr_epistemic_vec = torch.cat(tr_epistemic_vec).cpu().numpy()
    z_train = np.concatenate(z_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train


def latent_project_MNIST(
    BNN,
    VAE,
    dset,
    batch_size=1024,
    cuda=True,
    flatten_BNN=False,
    flatten_VAE=False,
    prob_BNN=True,
):
    if cuda:
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
        )
    else:
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=3
        )
    z_train = []
    y_train = []
    x_train = []
    tr_aleatoric_vec = []
    tr_epistemic_vec = []

    for j, (x, y_l) in enumerate(loader):

        if flatten_VAE:
            zz = VAE.recongnition(x.view(x.shape[0], -1)).loc.data.cpu().numpy()
        else:
            zz = VAE.recongnition(x).loc.data.cpu().numpy()

        if flatten_BNN:
            to_BNN = MNIST_mean_std_norm(x.view(x.shape[0], -1))
        else:
            to_BNN = MNIST_mean_std_norm(x)

        if prob_BNN:
            probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False).data
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(
                probs
            )
        else:
            probs = BNN.predict(to_BNN, grad=False)
            total_entropy = -(probs * torch.log(probs + 1e-10)).sum(
                dim=1, keepdim=False
            )
            aleatoric_entropy = total_entropy
            epistemic_entropy = total_entropy * 0

        tr_epistemic_vec.append(epistemic_entropy.data)
        tr_aleatoric_vec.append(aleatoric_entropy.data)

        z_train.append(zz)
        y_train.append(y_l.numpy())
        x_train.append(x.numpy())

    tr_aleatoric_vec = torch.cat(tr_aleatoric_vec).cpu().numpy()
    tr_epistemic_vec = torch.cat(tr_epistemic_vec).cpu().numpy()
    z_train = np.concatenate(z_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, lw=1):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    plt_return = ax.plot(x, y, color=color, lw=lw)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

    return plt_return


def gen_bar_plot(
    labels,
    data,
    title=None,
    xlabel=None,
    ylabel=None,
    probs=False,
    hor=False,
    save_file=None,
    max_fields=40,
    fs=7,
    verbose=False,
    sort=False,
    dpi=40,
    neg_color=True,
    ax=None,
    c=None,
):
    if c is None:
        c = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    c1 = c[0]
    c2 = c[1]  # ???

    add_tit = ""
    if data.shape[0] > max_fields or sort:
        if verbose:
            print(
                title
                + " Demasiados campos de datos, mostrando %d mas grandes" % max_fields
            )
        add_tit = " (top %d)" % max_fields
        abs_data = np.abs(data)
        sort_idx = np.flipud(np.argsort(abs_data))[:max_fields]
        labels = labels[sort_idx]
        data = data[sort_idx]

    if ax == None:
        plt.figure(dpi=dpi)
        ax = plt.gca()

    fst = 15

    if neg_color:
        c = np.array([c1] * data.shape[0])
        c[data < 0] = c2
    else:
        c = c1

    if hor:
        ax.barh(labels, data, 0.8, color=c)
        ax.invert_yaxis()
    else:
        ax.bar(labels, data, color=c)

    ax.xaxis.grid(alpha=0.35)
    ax.yaxis.grid(alpha=0.35)

    if title is not None:
        plt.title(title + add_tit)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if probs and hor:
        ax.set_xlim((0, 1))
    elif probs and not hor:
        ax.set_ylim((0, 1))

    ax.title.set_fontsize(fst)
    for item in (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(fs)
        item.set_weight("normal")
    ax.legend(prop={"size": fs, "weight": "normal"}, frameon=False)

    ax.autoscale(enable=True, axis="x", tight=True)
    ax.autoscale(enable=True, axis="y", tight=True)
    #     plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")

    return ax


def latent_map_2d_cat(BNN, VAE, vae_sig=True, steps=300, extent=4, batch_size=2048):
    """Returns numpy matrices for aleatoric and epistemic entropy sweeps of latent space around
    specified coordinates. Requieres Gaussian output VAE and BNN."""
    dim_range = np.linspace(-extent, extent, steps)
    dimx, dimy = np.meshgrid(dim_range, dim_range)
    dim_mtx = np.concatenate((np.expand_dims(dimx, 2), np.expand_dims(dimy, 2)), axis=2)

    z = torch.from_numpy(dim_mtx).type(torch.FloatTensor).cuda()
    z = z.view(steps ** 2, 2)
    z_mat = z.data.cpu().numpy()

    iterator = generate_ind_batch(z.shape[0], batch_size, random=False, roundup=True)

    entropy_vec = []
    aleatoric_vec = []
    epistemic_vec = []

    for idx_set in iterator:

        if vae_sig:
            x = VAE.regenerate(z[idx_set, :], grad=False).loc.cpu()
        else:
            x = VAE.regenerate(z[idx_set, :], grad=False).cpu()
        probs = BNN.sample_predict(x, 0, False)

        total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(
            probs
        )

        entropy_vec.append(total_entropy.data)
        aleatoric_vec.append(aleatoric_entropy.data)
        epistemic_vec.append(epistemic_entropy.data)

    entropy_vec = torch.cat(entropy_vec).view(steps, steps).cpu().numpy()
    aleatoric_vec = torch.cat(aleatoric_vec).view(steps, steps).cpu().numpy()
    epistemic_vec = torch.cat(epistemic_vec).view(steps, steps).cpu().numpy()

    return z_mat, entropy_vec, aleatoric_vec, epistemic_vec


def CLUE_viewer(
    Nim,
    x_init_batch,
    y_init_batch,
    x_vec,
    BNN,
    VAE_regen=False,
    img_progress=False,
    loss_curves=False,
    img_delta=True,
    tgets=np.array(range(10)),
    cost_vec=None,
    aleatoric_vec=None,
    epistemic_vec=None,
    dist_vec=None,
):
    print("True target: ", tgets[y_init_batch[Nim]])

    if VAE_regen:
        fig, axes = plt.subplots(1, 3, dpi=210)
        axes[0].imshow(x_init_batch[Nim, 0, :, :], cmap="gray")
        axes[1].imshow(x_vec[0, Nim, 0, :, :], cmap="gray")
        axes[2].imshow(x_vec[-1, Nim, 0, :, :], cmap="gray")

    if img_progress:
        plt.figure(dpi=140)
        dd = make_grid(torch.Tensor(x_vec[:, Nim, :, :, :]), nrow=10).numpy()
        fig = plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation="nearest")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    if loss_curves:
        fig, axes = plt.subplots(1, 3, dpi=150)
        axes[0].plot(cost_vec[:, Nim])
        axes[0].set_title("Cost")
        axes[0].set_xlabel("iterations")

        axes[1].plot((aleatoric_vec[:, Nim] + epistemic_vec[:, Nim]))
        axes[1].set_title("Total Entropy")
        axes[1].set_xlabel("iterations")

        axes[2].plot((dist_vec[:, Nim]))
        axes[2].set_title("Ln Cost")
        axes[2].set_xlabel("iterations")

        plt.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None
        )

    if img_delta:
        N_explain = Nim

        to_BNN = MNIST_mean_std_norm(
            torch.tensor(x_init_batch[N_explain, 0, :, :]).view(1, -1)
        )
        probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False)
        total_entropy, o_aleatoric_entropy, o_epistemic_entropy = decompose_entropy_cat(
            probs
        )
        #     print('original aleatoric: %2.3f epistemic %2.3f' % (o_aleatoric_entropy.item(), o_epistemic_entropy.item()))
        _, o_preds = probs.mean(dim=0).sort(dim=1, descending=True)
        #     print('original predictions', o_preds.cpu().numpy())

        fig, ax = plt.subplots(nrows=1, ncols=3, dpi=200)
        ax[0].imshow(1 - x_init_batch[N_explain, 0, :, :], cmap="gray")
        ax[0].set_title(
            "a %2.3f e %2.3f" % (o_aleatoric_entropy.item(), o_epistemic_entropy.item())
        )
        ax[0].set_xlabel(tgets[o_preds.cpu().numpy()[0].astype(int)])

        to_BNN = MNIST_mean_std_norm(
            torch.tensor(x_vec[-1, N_explain, 0, :, :]).view(1, -1)
        )
        probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False)
        total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(
            probs
        )
        #     print('explanation aleatoric: %2.3f epistemic %2.3f' % (aleatoric_entropy.item(), epistemic_entropy.item()))
        _, preds = probs.mean(dim=0).sort(dim=1, descending=True)
        #     print('original predictions', preds.cpu().numpy())

        ax[1].imshow(1 - x_vec[-1, N_explain, 0, :, :], cmap="gray")
        ax[1].set_title(
            "a %2.3f e %2.3f" % (aleatoric_entropy.item(), epistemic_entropy.item())
        )
        ax[1].set_xlabel(tgets[preds.cpu().numpy()[0].astype(int)])

        mask = x_vec[-1] - x_init_batch

        mask_neg = -mask[N_explain, 0, :, :]
        mask_neg[mask_neg < 1e-3] = 0
        mask_neg = np.repeat(np.expand_dims(mask_neg, axis=2), 4, axis=2) * 2
        mask_neg[:, :, 0:2] = 0

        mask_pos = mask[N_explain, 0, :, :]
        mask_pos[mask_pos < 1e-3] = 0
        mask_pos = np.repeat(np.expand_dims(mask_pos, axis=2), 4, axis=2) * 2
        mask_pos[:, :, 1:3] = 0

        ax[2].imshow(1 - x_init_batch[N_explain, 0, :, :], cmap="gray")
        ax[2].imshow(mask_pos, alpha=0.5)  # ,  vmin=0, vmax=1)
        ax[2].imshow(mask_neg, alpha=0.8)
        ax[2].set_title("Mask")

        plt.show()


def FIDO_viewer(Nim, x_init_batch, explainer, BNN, VAEAC, tgets=np.array(range(10))):
    x_im = torch.Tensor(x_init_batch[Nim, :, :, :])

    to_BNN = MNIST_mean_std_norm(x_im.view(x_im.shape[0], -1))
    probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False)
    total_entropy, o_aleatoric_entropy, o_epistemic_entropy = decompose_entropy_cat(
        probs
    )
    #     print('original aleatoric: %2.3f epistemic %2.3f' % (o_aleatoric_entropy.item(), o_epistemic_entropy.item()))
    _, o_preds = probs.mean(dim=0).sort(dim=1, descending=True)
    #     print('original predictions', o_preds.cpu().numpy())

    explanation, mask = explainer.mask_inpaint(
        x_init_batch, VAEAC, flatten_ims=True, test_dims=10, cat=True
    )

    fig, ax = plt.subplots(nrows=1, ncols=3, dpi=200)
    ax[0].imshow(1 - x_init_batch[Nim, 0, :, :], cmap="gray")
    ax[0].set_title(
        "a %2.3f e %2.3f" % (o_aleatoric_entropy.item(), o_epistemic_entropy.item())
    )
    ax[0].set_xlabel(tgets[o_preds.cpu().numpy()[0]])

    to_BNN = MNIST_mean_std_norm(explanation[Nim, :].unsqueeze(0))
    probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False)
    total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(probs)
    #     print('explained aleatoric: %2.3f epistemic %2.3f' % (aleatoric_entropy.item(), epistemic_entropy.item()))
    _, preds = probs.mean(dim=0).sort(dim=1, descending=True)
    #     print('original predictions', preds.cpu().numpy())

    ax[1].imshow(1 - explanation[Nim, :].view(28, 28).data.cpu().numpy(), cmap="gray")
    ax[1].set_title(
        "a %2.3f e %2.3f" % (aleatoric_entropy.item(), epistemic_entropy.item())
    )
    ax[1].set_xlabel(tgets[preds.cpu().numpy()[0]])

    mask_pos = np.repeat(
        np.expand_dims(mask[Nim, :].view(28, 28).data.cpu().numpy(), axis=2), 4, axis=2
    )
    mask_pos[:, :, 1:3] = 0

    ax[2].imshow(1 - x_init_batch[Nim, 0, :, :], cmap="gray")
    ax[2].imshow(mask_pos, alpha=0.3)
    ax[2].set_title("Mask")


def CLUE_viewer_ext(
    Nim,
    x_init,
    true_labels,
    x_explain,
    o_aleatoric_entropy,
    o_epistemic_entropy,
    o_preds,
    aleatoric_entropy,
    epistemic_entropy,
    preds,
    tgets=np.array(range(10)),
):
    print("True target: ", tgets[true_labels[Nim]])

    N_explain = Nim

    fig, ax = plt.subplots(nrows=1, ncols=3, dpi=200)
    ax[0].imshow(1 - x_init[N_explain, 0, :, :], cmap="gray")
    ax[0].set_title(
        "a %2.3f e %2.3f"
        % (o_aleatoric_entropy[N_explain], o_epistemic_entropy[N_explain])
    )
    ax[0].set_xlabel(tgets[o_preds[N_explain].astype(int)])

    ax[1].imshow(1 - x_explain[N_explain, 0, :, :], cmap="gray")
    ax[1].set_title(
        "a %2.3f e %2.3f" % (aleatoric_entropy[N_explain], epistemic_entropy[N_explain])
    )
    ax[1].set_xlabel(tgets[preds[N_explain].astype(int)])

    mask = x_explain - x_init

    mask_neg = -mask[N_explain, 0, :, :]
    mask_neg[mask_neg < 1e-3] = 0
    mask_neg = np.repeat(np.expand_dims(mask_neg, axis=2), 4, axis=2) * 2
    mask_neg[:, :, 0:2] = 0

    mask_pos = mask[N_explain, 0, :, :]
    mask_pos[mask_pos < 1e-3] = 0
    mask_pos = np.repeat(np.expand_dims(mask_pos, axis=2), 4, axis=2) * 2
    mask_pos[:, :, 1:3] = 0

    ax[2].imshow(1 - x_init[N_explain, 0, :, :], cmap="gray")
    ax[2].imshow(mask_pos, alpha=0.5)  # ,  vmin=0, vmax=1)
    ax[2].imshow(mask_neg, alpha=0.8)
    ax[2].set_title("Mask")

    plt.show()
