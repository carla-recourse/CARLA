import os
from typing import Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from carla.models.catalog.ANN_TF import AnnModel as ann_tf
from carla.models.catalog.ANN_TORCH import AnnModel as ann_torch
from carla.models.catalog.Linear_TF import LinearModel as linear_tf
from carla.models.catalog.Linear_TORCH import LinearModel as linear_torch


def train_model(
    catalog_model,
    x: pd.DataFrame,
    y: pd.DataFrame,
    learning_rate: float,
    epochs: int,
    batch_size: int,
):
    """

    Parameters
    ----------
    catalog_model: MLModelCatalog
        API for classifier
    x: pd.DataFrame
        features
    y: pd.DataFrame
        labels
    learning_rate: float
        Learning rate for the training.
    epochs: int
        Number of epochs to train on.
    batch_size: int
        Size of each batch

    Returns
    -------

    """
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    if catalog_model.backend == "tensorflow":
        if catalog_model.model_type == "linear":
            model = linear_tf(
                dim_input=x.shape[1],
                num_of_classes=len(pd.unique(y)),
                data_name=catalog_model.data.name,
            )  # type: Union[linear_tf, ann_tf]
        elif catalog_model.model_type == "ann":
            model = ann_tf(
                dim_input=x.shape[1],
                dim_hidden_layer1=18,
                dim_hidden_layer2=9,
                dim_output_layer=3,
                num_of_classes=len(pd.unique(y)),
                data_name=catalog_model.data.name,
            )
        else:
            raise ValueError("model type not recognized")
        model.build_train_save_model(
            x_train, y_train, x_test, y_test, epochs, batch_size
        )
        model = model.model
    elif catalog_model.backend == "pytorch":
        train_dataset = DataFrameDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataFrameDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if catalog_model.model_type == "linear":
            model = linear_torch(dim_input=x.shape[1], num_of_classes=len(pd.unique(y)))
        elif catalog_model.model_type == "ann":
            model = ann_torch(
                input_layer=x.shape[1],
                hidden_layer_1=18,
                hidden_layer_2=9,
                output_layer=3,
                num_of_classes=len(pd.unique(y)),
            )
        else:
            raise ValueError("model type not recognized")
        training_torch(
            model,
            train_loader,
            test_loader,
            learning_rate,
            epochs,
            catalog_model.data.name,
            catalog_model.model_type,
        )
    else:
        raise ValueError("model backend not recognized")

    return model


class DataFrameDataset(Dataset):
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.tensor(x.to_numpy(), dtype=torch.float32).to(device)
        self.Y_train = torch.tensor(y.to_numpy()).to(device)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


def training_torch(
    model,
    train_loader,
    test_loader,
    learning_rate,
    epochs,
    data_name,
    model_name,
    model_directory="saved_models",
):
    loaders = {"train": train_loader, "test": test_loader}

    # Use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define the loss
    criterion = nn.BCELoss()

    # declaring optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # training
    # loss_per_iter = []
    # loss_per_batch = []
    trace_input = True
    for e in range(epochs):
        print("Epoch {}/{}".format(e, epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:

            running_loss = 0.0
            running_corrects = 0.0

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            for i, (inputs, labels) in enumerate(loaders[phase]):
                inputs = inputs.to(device)
                if trace_input:
                    x_for_trace = inputs
                    trace_input = False
                labels = labels.to(device)

                labels = labels.to(device).type(torch.int64)
                labels = torch.nn.functional.one_hot(labels)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    torch.argmax(outputs, axis=1)
                    == torch.argmax(labels, axis=1).float()
                )

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            print()

    # save model
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    model_to_trace = model.to("cpu")
    traced = torch.jit.trace(model_to_trace, (x_for_trace.cpu().float()))
    traced.save(
        f"{model_directory}/{model_name}_{data_name}_input_{model.input_neurons}.pt",
    )
