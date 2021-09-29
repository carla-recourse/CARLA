import numpy as np

from carla.data.causal_model import CausalModel

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


def _remove_prefix(node):
    """replaces e.g. x101 or u101 with 101"""
    assert node[0] == "x" or node[0] == "u"
    return node[1:]


def _load_scm_equations(scm_class: str):

    ###########################
    #  loading scm equations  #
    ###########################
    if scm_class == "sanity-3-lin":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = sanity_3_lin()
    elif scm_class == "sanity-3-anm":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = sanity_3_anm()
    elif scm_class == "_bu_sanity-3-gen":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = bu_sanity_3_gen()
    elif scm_class == "sanity-3-gen-OLD":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = sanity_3_gen_old()
    elif scm_class == "sanity-3-gen-NEW":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = sanity_3_gen_new()
    elif scm_class == "sanity-3-gen":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = sanity_3_gen()
    elif scm_class == "sanity-6-lin":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = sanity_6_lin()
    elif scm_class == "german-credit":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = german_credit()
    elif scm_class == "adult":
        structural_equations_np, structural_equations_ts, noise_distributions = adult()
    elif scm_class == "fair-IMF-LIN" or scm_class == "fair-IMF-LIN-radial":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = fair_imf_lin()
    elif scm_class == "fair-CAU-LIN" or scm_class == "fair-CAU-LIN-radial":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = fair_cau_lin()
    elif scm_class == "fair-CAU-ANM" or scm_class == "fair-CAU-ANM-radial":
        (
            structural_equations_np,
            structural_equations_ts,
            noise_distributions,
        ) = fair_cau_anm()
    else:
        raise Exception(f"scm_class `{scm_class}` not recognized.")

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


def load_scm(scm_class: str):
    """

    Parameters
    ----------
    scm_class: name of the structural equations

    Returns
    -------
    CausalModel
    """

    (
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
    ) = _load_scm_equations(scm_class)

    scm = CausalModel(
        scm_class,
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
    )

    return scm
