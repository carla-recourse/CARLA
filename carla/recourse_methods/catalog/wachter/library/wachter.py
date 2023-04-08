import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

from carla import log
from carla.recourse_methods.processing import reconstruct_encoding_constraints

DECISION_THRESHOLD = 0.5


def wachter_recourse(
    torch_model,
    x: np.ndarray,
    cat_feature_indices: List[int],
    binary_cat_features: bool,
    feature_costs: Optional[List[float]],
    lr: float,
    lambda_param: float,
    y_target: List[int],
    n_iter: int,
    t_max_min: float,
    norm: int,
    clamp: bool,
    loss_type: str,
) -> np.ndarray:
    """
    Generates counterfactual example according to Wachter et.al for input instance x

    Parameters
    ----------
    torch_model:
        black-box-model to discover
    x:
        Factual instance to explain.
    cat_feature_indices:
        List of positions of categorical features in x.
    binary_cat_features:
        If true, the encoding of x is done by drop_if_binary.
    feature_costs:
        List with costs per feature.
    lr:
        Learning rate for gradient descent.
    lambda_param:
        Weight factor for feature_cost.
    y_target:
        Tuple of class probabilities (BCE loss) or [Float] for logit score (MSE loss).
    n_iter:
        Maximum number of iterations.
    t_max_min:
        Maximum time amount of search.
    norm:
        L-norm to calculate cost.
    clamp:
        If true, feature values will be clamped to intverval [0, 1].
    loss_type:
        String for loss function ("MSE" or "BCE").

    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # returns counterfactual instance
    torch.manual_seed(0)

    if feature_costs is not None:
        feature_costs = torch.from_numpy(feature_costs).float().to(device)

    x = torch.from_numpy(x).float().to(device)
    y_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)
    # x_new is used for gradient search in optimizing process
    x_new = Variable(x.clone(), requires_grad=True)
    # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
    # such that categorical data is either 0 or 1
    x_new_enc = reconstruct_encoding_constraints(
        x_new, cat_feature_indices, binary_cat_features
    )

    optimizer = optim.Adam([x_new], lr, amsgrad=True)

    if loss_type == "MSE":
        if len(y_target) != 1:
            raise ValueError(f"y_target {y_target} is not a single logit score")

        # If logit is above 0.0 we want class 1, else class 0
        target_class = int(y_target[0] > 0.0)
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "BCE":
        if y_target[0] + y_target[1] != 1.0:
            raise ValueError(
                f"y_target {y_target} does not contain 2 valid class probabilities"
            )

        # [0, 1] for class 1, [1, 0] for class 0
        # target is the class probability of class 1
        # target_class is the class with the highest probability
        target_class = torch.round(y_target[1]).int()
        loss_fn = torch.nn.BCELoss()
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    # get the probablity of the target class
    f_x_new = torch_model(x_new)[:, target_class]

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)
    while f_x_new <= DECISION_THRESHOLD:
        it = 0
        while f_x_new <= DECISION_THRESHOLD and it < n_iter:
            optimizer.zero_grad()
            x_new_enc = reconstruct_encoding_constraints(
                x_new, cat_feature_indices, binary_cat_features
            )
            # use x_new_enc for prediction results to ensure constraints
            # get the probablity of the target class
            f_x_new = torch_model(x_new_enc)[:, target_class]

            if loss_type == "MSE":
                # single logit score for the target class for MSE loss
                f_x_loss = torch.log(f_x_new / (1 - f_x_new))
            elif loss_type == "BCE":
                # tuple output for BCE loss
                f_x_loss = torch_model(x_new_enc).squeeze(axis=0)
            else:
                raise ValueError(f"loss_type {loss_type} not supported")

            cost = (
                torch.dist(x_new_enc, x, norm)
                if feature_costs is None
                else torch.norm(feature_costs * (x_new_enc - x), norm)
            )

            loss = loss_fn(f_x_loss, y_target) + lamb * cost
            loss.backward()
            optimizer.step()
            # clamp potential CF
            if clamp:
                x_new.clone().clamp_(0, 1)
            it += 1
        lamb -= 0.05

        if datetime.datetime.now() - t0 > t_max:
            break
    return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
