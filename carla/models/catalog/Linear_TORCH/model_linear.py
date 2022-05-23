import numpy as np
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, dim_input, num_of_classes):
        """

        Parameters
        ----------
        dim_input: int > 0
            number of neurons for this layer
        num_of_classes: int > 0
            number of classes
        """
        super().__init__()

        # number of input neurons
        self.input_neurons = dim_input

        # Layer
        self.output = nn.Linear(dim_input, num_of_classes)

        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Forward pass through the network
        :param input: tabular data
        :return: prediction
        """
        output = self.output(x)
        output = self.softmax(output)

        # output = output.squeeze()

        return output

    def predict(self, data):
        """
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        return self.forward(input).detach().numpy()
