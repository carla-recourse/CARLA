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
    binary_cat_features: bool = True,
    feature_costs: Optional[List[float]] = None,
    lr: float = 0.01,
    lambda_param: float = 0.01,
    y_target: List[int] = [0, 1],
    n_iter: int = 1000,
    t_max_min: float = 0.5,
    norm: int = 1,
    clamp: bool = True,
    loss_type: str = "MSE",
) -> np.ndarray:
    """
    Generates counterfactual example according to Wachter et.al for input instance x

    Parameters
    ----------
    torch_model: black-box-model to discover
    x: factual to explain
    cat_feature_indices: list of positions of categorical features in x
    binary_cat_features: If true, the encoding of x is done by drop_if_binary
    feature_costs: List with costs per feature
    lr: learning rate for gradient descent
    lambda_param: weight factor for feature_cost
    y_target: List of one-hot-encoded target class
    n_iter: maximum number of iteration
    t_max_min: maximum time of search
    norm: L-norm to calculate cost
    clamp: If true, feature values will be clamped to (0, 1)
    loss_type: String for loss function (MSE or BCE)

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
    softmax = nn.Softmax()

    if loss_type == "MSE":
        loss_fn = torch.nn.MSELoss()
        f_x_new = softmax(torch_model(x_new))[1]
    else:
        loss_fn = torch.nn.BCELoss()
        f_x_new = torch_model(x_new)[:, 1]

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)

    while f_x_new <= DECISION_THRESHOLD:
        it = 0
        while f_x_new <= 0.5 and it < n_iter:
            optimizer.zero_grad()
            x_new_enc = reconstruct_encoding_constraints(
                x_new, cat_feature_indices, binary_cat_features
            )
            # use x_new_enc for prediction results to ensure constraints
            f_x_new = softmax(torch_model(x_new_enc))[:, 1]
            f_x_new_binary = torch_model(x_new_enc).squeeze(axis=0)

            cost = (
                torch.dist(x_new_enc, x, norm)
                if feature_costs is None
                else torch.norm(feature_costs * (x_new_enc - x), norm)
            )

            loss = loss_fn(f_x_new_binary, y_target) + lamb * cost
            loss.backward()
            optimizer.step()
            # clamp potential CF
            if clamp:
                x_new.clone().clamp_(0, 1)
            it += 1
        lamb -= 0.05

        if datetime.datetime.now() - t0 > t_max:
            log.info("Timeout - No Counterfactual Explanation Found")
            break
        elif f_x_new >= 0.5:
            log.info("Counterfactual Explanation Found")
    return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
