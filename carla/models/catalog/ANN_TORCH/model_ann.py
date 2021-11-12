import numpy as np
import torch
from torch import nn


class AnnModel(nn.Module):
    def __init__(self, input_layer, hidden_layers, num_of_classes):
        """
        Defines the structure of the neural network

        Parameters
        ----------
        input_layer: int > 0
            Dimension of the input / number of features
        hidden_layers: list
            List where each element is the number of neurons in the ith hidden layer
        num_of_classes: int > 0
            Dimension of the output / number of classes.
        """
        super().__init__()

        # Layer
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_layer, hidden_layers[0]))
        # hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        # output layer
        self.layers.append(nn.Linear(hidden_layers[-1], num_of_classes))

        # self.input = nn.Linear(input_layer, hidden_layer_1)
        # self.hidden_1 = nn.Linear(hidden_layer_1, hidden_layer_2)
        # self.hidden_2 = nn.Linear(hidden_layer_2, output_layer)
        # self.output = nn.Linear(output_layer, num_of_classes)

        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Forward pass through the network

        Parameters
        ----------
        x: tabular data
            input

        Returns
        -------
        prediction
        """
        # output = self.input(x)
        # output = self.relu(output)
        # output = self.hidden_1(output)
        # output = self.relu(output)
        # output = self.hidden_2(output)
        # output = self.relu(output)
        # output = self.output(output)
        # output = self.softmax(output)

        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
            else:
                x = self.softmax(x)

        return x

    def proba(self, data):
        """
        Computes probabilistic output for two classes

        Parameters
        ----------
        data: torch tabular
            input

        Returns
        -------
        np.array

        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        class_1 = 1 - self.forward(input)
        class_2 = self.forward(input)

        return list(zip(class_1, class_2))

    def prob_predict(self, data):
        """
        Computes probabilistic output for two classes

        Parameters
        ----------
        data: torch tabular
            input

        Returns
        -------
        np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data)

        class_1 = 1 - self.forward(input).detach().numpy().squeeze()
        class_2 = self.forward(input).detach().numpy().squeeze()

        # For single prob prediction it happens, that class_1 is casted into float after 1 - prediction
        # Additionally class_1 and class_2 have to be at least shape 1
        if not isinstance(class_1, np.ndarray):
            class_1 = np.array(class_1).reshape(1)
            class_2 = class_2.reshape(1)

        return np.array(list(zip(class_1, class_2)))

    def predict(self, data):
        """
        predict method for CFE-Models which need this method.

        Parameters
        ----------
        data: Union(torch, list)

        Returns
        -------
        np.array with prediction

        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        return self.forward(input).detach().numpy()
