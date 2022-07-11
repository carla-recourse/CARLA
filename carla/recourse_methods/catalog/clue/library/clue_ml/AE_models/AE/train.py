from __future__ import division, print_function

import time

import torch.utils.data
from numpy.random import normal
from torchvision.utils import make_grid

from carla.recourse_methods.catalog.clue.library.clue_ml.src.utils import *


def train_VAE(
    net,
    name,
    batch_size,
    nb_epochs,
    trainset,
    valset,
    cuda,
    flat_ims=False,
    train_plot=False,
    Nclass=None,
    early_stop=None,
    script_mode=False,
):

    models_dir = name
    results_dir = name + "_results"

    if cuda:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
        )

    else:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )

    ## ---------------------------------------------------------------------------------------------------------------------
    # net dims
    log.info("\nNetwork:")

    epoch = 0

    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    log.info("\nTrain:")

    log.info("init cost variables:")
    vlb_train = np.zeros(nb_epochs)
    vlb_dev = np.zeros(nb_epochs)
    best_vlb = -np.inf
    best_vlb_train = -np.inf
    best_epoch = 0

    nb_its_dev = 1

    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)

        tic = time.time()
        nb_samples = 0
        for x, y in trainloader:

            if flat_ims:
                x = x.view(x.shape[0], -1)
            if Nclass is not None:
                y_oh = torch_onehot(y, Nclass).type(x.type())
                x = torch.cat([x, y_oh], 1)

            cost, _ = net.fit(x)

            vlb_train[i] += cost * len(x)
            nb_samples += len(x)

        vlb_train[i] /= nb_samples

        toc = time.time()

        # ---- print
        log.info("it %d/%d, vlb %f, " % (i, nb_epochs, vlb_train[i]))
        log.info("time: %f seconds\n" % (toc - tic))
        net.update_lr(i)

        if vlb_train[i] > best_vlb_train:
            best_vlb_train = vlb_train[i]

        # ---- dev
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):

                if flat_ims:
                    x = x.view(x.shape[0], -1)
                if Nclass is not None:
                    y_oh = torch_onehot(y, Nclass).type(x.type())
                    x = torch.cat([x, y_oh], 1)

                cost, _ = net.eval(x)

                vlb_dev[i] += cost * len(x)
                nb_samples += len(x)

            vlb_dev[i] /= nb_samples

            log.info("vlb %f (%f)\n" % (vlb_dev[i], best_vlb))

            if train_plot:
                zz = net.recongnition(x).sample()
                o = net.regenerate(zz)
                try:
                    o = o.cpu()
                except:
                    o = o.loc.cpu()
                if len(x.shape) == 2:
                    side = int(np.sqrt(x.shape[1]))
                    x = x.view(-1, 1, side, side).data
                    o = o.view(-1, 1, side, side).data

                import matplotlib.pyplot as plt

                plt.figure()
                dd = make_grid(torch.cat([x[:10], o[:10]]), nrow=10).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation="nearest")
                if script_mode:
                    plt.savefig(results_dir + "/rec%d.png" % i)
                else:
                    plt.show()

                z_sample = normal(loc=0.0, scale=1.0, size=(36, net.latent_dim))
                x_rec = net.regenerate(z_sample)
                try:
                    x_rec = x_rec.cpu()
                except:
                    x_rec = x_rec.loc.cpu()
                if len(x_rec.shape) == 2:
                    side = int(np.sqrt(x_rec.shape[1]))
                    x_rec = x_rec.view(-1, 1, side, side)
                plt.figure()
                dd = make_grid(x_rec, nrow=6).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation="nearest")
                if script_mode:
                    plt.savefig(results_dir + "/sample%d.png" % i)
                else:
                    plt.show()

        if vlb_dev[i] > best_vlb:
            best_vlb = vlb_dev[i]
            best_epoch = i
            net.save(os.path.join(models_dir, "theta_best.dat"))

        if early_stop is not None and (i - best_epoch) > early_stop:
            break

    net.save(models_dir + "/theta_last.dat")
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    log.info("average time: %f seconds\n" % runtime_per_it)

    ## ---------------------------------------------------------------------------------------------------------------------
    # results
    log.info("\nRESULTS:")
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = best_vlb
    best_cost_train = best_vlb_train

    log.info("best_vlb_dev: %f" % best_cost_dev)
    log.info("best_vlb_train: %f" % best_cost_train)
    log.info("nb_parameters: %d (%s)\n" % (nb_parameters, humansize(nb_parameters)))

    ## ---------------------------------------------------------------------------------------------------------------------
    return vlb_train, vlb_dev
