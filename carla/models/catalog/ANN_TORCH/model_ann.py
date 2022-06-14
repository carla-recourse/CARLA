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

        self.input_neurons = input_layer

        # Layer
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_layer, hidden_layers[0]))
        # hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        # output layer
        self.layers.append(nn.Linear(hidden_layers[-1], num_of_classes))

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
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
            else:
                x = self.softmax(x)

        return x

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
