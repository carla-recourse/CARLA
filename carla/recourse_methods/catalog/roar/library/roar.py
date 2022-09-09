import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from scipy.optimize import linprog
from torch import nn
from torch.autograd import Variable, grad

from carla import log
from carla.recourse_methods.processing import reconstruct_encoding_constraints


def _calc_max_perturbation(
    recourse: torch.Tensor,
    coeff: torch.Tensor,
    intercept: torch.Tensor,
    delta_max: float,
    target_class: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the optimal delta perturbation using linear program
    Adjusted from: https://github.com/AI4LIFE-GROUP/ROAR/blob/main/recourse_methods.py

    Parameters
    ----------
    recourse: Factual to explain
    coeff: Tensor with Coefficients of linear model
    intercept: Tensor of Intercept of linear model
    delta_max: Maximum perturbations of weights
    target_class: Float Tensor representing the target class

    Returns
    -------
    Tuple of Torch tensors with optimal perturbations of coefficients and intercept
    """
    W = torch.cat((coeff, intercept), 0)  # Add intercept to weights
    recourse = torch.cat(
        (recourse, torch.ones(1, device=recourse.device)), 0
    )  # Add 1 to the feature vector for intercept

    loss_fn = torch.nn.BCELoss()
    W.requires_grad = True
    f_x_new = torch.nn.Sigmoid()(torch.matmul(W, recourse))
    w_loss = loss_fn(f_x_new, target_class)

    gradient_w_loss = grad(w_loss, W)[0]
    c = list(np.array(gradient_w_loss.cpu()) * np.array([-1] * len(gradient_w_loss)))
    bound = (-delta_max, delta_max)
    bounds = [bound] * len(gradient_w_loss)

    res = linprog(c, bounds=bounds, method="simplex")

    if res.status != 0:
        log.warning("Optimization with respect to delta failed to converge")

    delta_opt = res.x
    delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])

    return delta_W, delta_W0


def roar_recourse(
    torch_model,
    x: np.ndarray,
    coeff: np.ndarray,
    intercept: np.float,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    feature_costs: Optional[List[float]] = None,
    lr: float = 0.01,
    lambda_param: float = 0.01,
    delta_max: float = 0.01,
    y_target: List[int] = [0, 1],
    t_max_min: float = 0.5,
    norm: int = 1,
    loss_type: str = "BCE",
    loss_threshold: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """
    Generates counterfactual example according to ROAR method for input instance x

    Parameters
    ----------
    torch_model: carla.model.MLModel.TorchModel
        Black-box-model to discover.
    x: np.ndarray
        Factual to explain.
    coeff: np.ndarray
        Coefficient for factual x.
    intercept: np.float
        Intercept for factual x.
    cat_feature_indices: list
        List of positions of categorical features in x.
    binary_cat_features: bool
        If true, the encoding of x is done by drop_if_binary.
    feature_cost: list
        List with costs per feature.
    lr: float
        Learning rate for gradient descent.
    lambda_param: float
        Weight factor for feature_cost.
    delta_max: float
        Maximum perturbation for weights.
    y_target: list
        List of one-hot-encoded target class.
    t_max_min: float
        Maximum time of search.
    norm: int
        L-norm to calculate cost.
    loss_type: String
        String for loss function. "MSE" or "BCE".
    loss_threshold: float
        Threshold for loss difference
    seed: int
        Seed for torch when generating counterfactuals.

    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    if feature_costs is not None:
        feature_costs = torch.from_numpy(feature_costs).float().to(device)

    coeff = torch.from_numpy(coeff).float().to(device)
    intercept = torch.from_numpy(np.asarray([intercept])).float().to(device)
    x = torch.from_numpy(x).float().to(device)
    y_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)

    # x_new is used for gradient search in optimizing process
    x_new = Variable(x.clone(), requires_grad=True)

    optimizer = optim.Adam([x_new], lr=lr, amsgrad=True)

    if loss_type == "MSE":
        if len(y_target) != 1:
            raise ValueError(f"y_target {y_target} is not a single logit score")

        # If logit is above 0.0 we want class 1, else class 0
        target_class = torch.tensor(y_target[0] > 0.0).float()
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "BCE":
        if y_target[0] + y_target[1] != 1.0:
            raise ValueError(
                f"y_target {y_target} does not contain 2 valid class probabilities"
            )

        # [0, 1] for class 1, [1, 0] for class 0
        # target is the class probability of class 1
        target_class = torch.round(y_target[1]).float()
        loss_fn = torch.nn.BCELoss()
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    # Placeholder values for first loop
    loss = torch.tensor(0)
    loss_diff = loss_threshold + 1

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)

    while loss_diff > loss_threshold:
        loss_prev = loss.clone().detach()

        # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
        # such that categorical data is either 0 or 1
        x_new_enc = reconstruct_encoding_constraints(
            x_new, cat_feature_indices, binary_cat_features
        )

        # Calculate max delta perturbation on weights
        delta_W, delta_W0 = _calc_max_perturbation(
            x_new_enc.squeeze(), coeff, intercept, delta_max, target_class
        )
        delta_W, delta_W0 = (
            torch.from_numpy(delta_W).float().to(device),
            torch.from_numpy(delta_W0).float().to(device),
        )

        optimizer.zero_grad()

        # get the probability of the target class
        f_x_new = torch.nn.Sigmoid()(
            torch.matmul(coeff + delta_W, x_new_enc.squeeze()) + intercept + delta_W0
        ).squeeze()

        if loss_type == "MSE":
            # single logit score for the target class for MSE loss
            f_x_new = torch.log(f_x_new / (1 - f_x_new))

        cost = (
            torch.dist(x_new_enc, x, norm)
            if feature_costs is None
            else torch.norm(feature_costs * (x_new_enc - x), norm)
        )

        loss = loss_fn(f_x_new, target_class) + lamb * cost
        loss.backward()

        optimizer.step()

        loss_diff = torch.dist(loss_prev, loss, 2)

        if datetime.datetime.now() - t0 > t_max:
            log.info("Timeout - ROAR didn't converge")
            break

    return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
