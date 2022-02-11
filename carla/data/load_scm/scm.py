"""
Based on https://github.com/amirhk/recourse/blob/master/loadSCM.py
"""
from carla.data.load_scm.distributions import MixtureOfGaussians, Normal


def sanity_3_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -x1 + n_samples,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
    }
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical = []
    immutables = []

    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )
