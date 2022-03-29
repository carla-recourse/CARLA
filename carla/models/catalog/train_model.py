from typing import Union

import pandas as pd
import torch
import xgboost
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset

from carla.models.catalog.ANN_TF import AnnModel
from carla.models.catalog.ANN_TF import AnnModel as ann_tf
from carla.models.catalog.ANN_TORCH import AnnModel as ann_torch
from carla.models.catalog.Linear_TF import LinearModel
from carla.models.catalog.Linear_TF import LinearModel as linear_tf
from carla.models.catalog.Linear_TORCH import LinearModel as linear_torch


def train_model(
    catalog_model,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    hidden_size: list,
    n_estimators: int,
    max_depth: int,
) -> Union[LinearModel, AnnModel, RandomForestClassifier, xgboost.XGBClassifier]:
    """

    Parameters
    ----------
    catalog_model: MLModelCatalog
        API for classifier
    x_train: pd.DataFrame
        training features
    y_train: pd.DataFrame
        training labels
    x_test: pd.DataFrame
        test features
    y_test: pd.DataFrame
        test labels
    learning_rate: float
        Learning rate for the training.
    epochs: int
        Number of epochs to train on.
    batch_size: int
        Size of each batch
    hidden_size: list[int]
        hidden_size[i] contains the number of nodes in layer [i].
    n_estimators: int
        Number of trees in forest
    max_depth: int
        Max depth of trees in forest

    Returns
    -------
    Union[LinearModel, AnnModel, RandomForestClassifier, xgboost.XGBClassifier]
    """
    print(f"balance on test set {y_train.mean()}, balance on test set {y_test.mean()}")
    if catalog_model.backend == "tensorflow":
        if catalog_model.model_type == "linear":
            model = linear_tf(
                dim_input=x_train.shape[1],
                num_of_classes=len(pd.unique(y_train)),
                data_name=catalog_model.data.name,
            )  # type: Union[linear_tf, ann_tf]
        elif catalog_model.model_type == "ann":
            model = ann_tf(
                dim_input=x_train.shape[1],
                dim_hidden_layers=hidden_size,
                num_of_classes=len(pd.unique(y_train)),
                data_name=catalog_model.data.name,
            )
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
        model.build_train_save_model(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs,
            batch_size,
            model_name=catalog_model.model_type,
        )
        return model.model
    elif catalog_model.backend == "pytorch":
        train_dataset = DataFrameDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataFrameDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if catalog_model.model_type == "linear":
            model = linear_torch(
                dim_input=x_train.shape[1], num_of_classes=len(pd.unique(y_train))
            )
        elif catalog_model.model_type == "ann":
            model = ann_torch(
                input_layer=x_train.shape[1],
                hidden_layers=hidden_size,
                num_of_classes=len(pd.unique(y_train)),
            )
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
        _training_torch(
            model,
            train_loader,
            test_loader,
            learning_rate,
            epochs,
        )
        return model
    elif catalog_model.backend == "sklearn":
        if catalog_model.model_type == "forest":
            random_forest_model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth
            )
            random_forest_model.fit(X=x_train, y=y_train)
            train_score = random_forest_model.score(X=x_train, y=y_train)
            test_score = random_forest_model.score(X=x_test, y=y_test)
            print(
                "model fitted with training score {} and test score {}".format(
                    train_score, test_score
                )
            )
            return random_forest_model
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
    elif catalog_model.backend == "xgboost":
        if catalog_model.model_type == "forest":
            param = {
                "max_depth": max_depth,
                "objective": "binary:logistic",
                "n_estimators": n_estimators,
            }
            xgboost_model = xgboost.XGBClassifier(**param)
            xgboost_model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                eval_metric="logloss",
                verbose=True,
            )
            return xgboost_model
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
    else:
        raise ValueError("model backend not recognized")


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


def _training_torch(
    model,
    train_loader,
    test_loader,
    learning_rate,
    epochs,
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
                labels = labels.to(device).type(torch.int64)
                labels = torch.nn.functional.one_hot(labels, num_classes=2)

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
