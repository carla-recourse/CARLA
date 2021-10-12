import numpy as np

from .scm import (
    adult,
    bu_sanity_3_gen,
    fair_cau_anm,
    fair_cau_lin,
    fair_imf_lin,
    german_credit,
    sanity_3_anm,
    sanity_3_gen,
    sanity_3_gen_new,
    sanity_3_gen_old,
    sanity_3_lin,
    sanity_6_lin,
)

scm_dict = {
    "adult": adult,
    "german-credit": german_credit,
    "sanity-3-anm": sanity_3_anm,
    "sanity-3-lin": sanity_3_lin,
    "sanity-6-lin": sanity_6_lin,
    "sanity-3-gen": sanity_3_gen,
    "sanity-3-gen-OLD": sanity_3_gen_old,
    "sanity-3-gen-NEW": sanity_3_gen_new,
    "_bu_sanity-3-gen": bu_sanity_3_gen,
    "fair-IMF-LIN": fair_imf_lin,
    "fair-CAU-LIN": fair_cau_lin,
    "fair-CAU-ANM": fair_cau_anm,
    "fair-IMF-LIN-radial": fair_imf_lin,
    "fair-CAU-LIN-radial": fair_cau_lin,
    "fair-CAU-ANM-radial": fair_cau_anm,
}


def _remove_prefix(node):
    """replaces e.g. x101 or u101 with 101"""
    assert node[0] == "x" or node[0] == "u"
    return node[1:]


def load_scm_equations(scm_class: str):
    ###########################
    #  loading scm equations  #
    ###########################
    structural_equations_np, structural_equations_ts, noise_distributions = scm_dict[
        scm_class
    ]()

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

    return structural_equations_np, structural_equations_ts, noise_distributions
