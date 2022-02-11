import numpy as np

from .scm import sanity_3_lin

scm_dict = {
    "sanity-3-lin": sanity_3_lin,
}


def _remove_prefix(node):
    """replaces e.g. x101 or u101 with 101"""
    assert node[0] == "x" or node[0] == "u"
    return node[1:]


def load_scm_equations(scm_class: str):
    ###########################
    #  loading scm equations  #
    ###########################
    (
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
        continuous,
        categorical,
        immutables,
    ) = scm_dict[scm_class]()

    ###########################
    #       some checks       #
    ###########################
    # TODO duplicate with tests
    if not (
        [_remove_prefix(node) for node in structural_equations_np.keys()]
        == [_remove_prefix(node) for node in structural_equations_ts.keys()]
        == [_remove_prefix(node) for node in noise_distributions.keys()]
    ):
        raise ValueError(
            "structural_equations_np & structural_equations_ts & noises_distributions should have identical keys."
        )

    if not (
        np.all(["x" in node for node in structural_equations_np.keys()])
        and np.all(["x" in node for node in structural_equations_ts.keys()])
    ):
        raise ValueError("endogenous variables must start with `x`.")

    return (
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
        continuous,
        categorical,
        immutables,
    )
