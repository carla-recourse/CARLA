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

    # TODO what is this for
    # structural_equations_np = {
    #   'x1': lambda n_samples, : n_samples,
    #   # 'x2': TBD
    # }
    # structural_equations_ts = {
    #   'x1': lambda n_samples, : n_samples,
    #   # 'x2': TBD
    # }
    # noises_distributions = {
    #   'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
    #   'u2': Normal(0, 1),
    # }

    # if scm_class == 'sanity-2-add':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : 2 * x1 + n_samples
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : 2 * x1 + n_samples
    # elif scm_class == 'sanity-2-mult':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : x1 * n_samples
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : x1 * n_samples
    # elif scm_class == 'sanity-2-add-pow':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : (x1 + n_samples) ** 2
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : (x1 + n_samples) ** 2
    # elif scm_class == 'sanity-2-add-sig':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : 5 / (1 + np.exp(- x1 - n_samples))
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : 5 / (1 + torch.exp(- x1 - n_samples))
    # elif scm_class == 'sanity-2-sig-add':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : 5 / (1 + np.exp(-x1)) + n_samples
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : 5 / (1 + torch.exp(-x1)) + n_samples
    # elif scm_class == 'sanity-2-pow-add':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : x1 ** 2 + n_samples
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : x1 ** 2 + n_samples
    # elif scm_class == 'sanity-2-sin-add':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : np.sin(x1) + n_samples
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : torch.sin(x1) + n_samples
    # elif scm_class == 'sanity-2-cos-exp-add':
    #   structural_equations_np['x2'] = lambda n_samples, x1 : 2 * np.cos(3 * x1) * np.exp(-0.3 * x1**2) + n_samples
    #   structural_equations_ts['x2'] = lambda n_samples, x1 : 2 * torch.cos(3 * x1) * torch.exp(-0.3 * x1**2) + n_samples

    # ============================================================================
    # ABOVE: 2-variable sanity models used for checking cond. dist. fit
    # BELOW: 3+variable sanity models used in paper
    # ============================================================================

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
