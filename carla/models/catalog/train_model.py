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
    **kws,
):
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
            x_train, y_train, x_test, y_test, learning_rate, epochs, batch_size
        )
        model = model.model
    elif catalog_model.backend == "pytorch":
        train_dataset = DataFrameDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataFrameDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if catalog_model.model_type == "linear":
            model = linear_torch(
                input_layer=x.shape[1], output_layer=2, num_of_classes=len(pd.unique(y))
            )
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
    def weighted_binary_cross_entropy(output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2

            loss = weights[1] * (target * torch.log(output)) + weights[0] * (
                (1 - target) * torch.log(1 - output)
            )
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))

    # Use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define the loss
    criterion = nn.BCELoss()

    # declaring optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # training
    loss_per_iter = []
    loss_per_batch = []
    for e in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.nn.functional.one_hot(labels)

            # Training pass
            optimizer.zero_grad()

            outputs = model(inputs.float())

            """
            Due to high class imbalance for give-me credit data set, we give 0-class higher weight
            For the ANN, the weight has to be quite high; otherwise constant class 1 is predicted
            """
            if data_name == "give-me":
                loss = weighted_binary_cross_entropy(
                    outputs.reshape(-1), labels.float(), [0.7, 0.3]
                )
            else:
                loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        print("Epoch {} Loss {:.5f}".format(e, running_loss / (i + 1)))

    # Comparing training to test
    dataiter = iter(test_loader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = torch.nn.functional.one_hot(labels)
    outputs = model(inputs.float())
    print("==========================================")
    print("Binary Cross Entropy loss")
    train_loss = loss_per_batch[-1]
    print("Training:", train_loss)
    test_loss = (
        criterion(
            outputs,
            labels.float(),
        )
        .detach()
        .cpu()
        .numpy()
    )
    print("Test:", test_loss)

    # save model
    torch.save(
        model.state_dict(),
        f"{model_directory}/{model_name}_{data_name}_input_{model.input_neurons}.pt",
    )
