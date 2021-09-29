import numpy as np
import torch

from carla.data.load_scm.distributions import (
    Bernoulli,
    Gamma,
    MixtureOfGaussians,
    Normal,
)


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

    return structural_equations_np, structural_equations_ts, noises_distributions


def sanity_3_anm():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: 3 / (1 + np.exp(-2.0 * x1)) - 1 + n_samples,
        "x3": lambda n_samples, x1, x2: -0.5 * (0.1 * x1 + 0.5 * x2 ** 2) + n_samples,
    }
    structural_equations_ts = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: 3 / (1 + torch.exp(-2.0 * x1)) - 1 + n_samples,
        "x3": lambda n_samples, x1, x2: -0.5 * (0.1 * x1 + 0.5 * x2 ** 2) + n_samples,
    }
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 0.1),
        "u3": Normal(0, 1),
    }

    return structural_equations_np, structural_equations_ts, noises_distributions


def bu_sanity_3_gen():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -3
        * (1 / (1 + np.exp(-2.0 * x1 + n_samples)) - 0.4),
        "x3": lambda n_samples, x1, x2: -0.5
        * (0.1 * x1 + 0.5 * (x2 - 0.0) ** 2 * n_samples),
    }
    structural_equations_ts = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -3
        * (1 / (1 + torch.exp(-2.0 * x1 + n_samples)) - 0.4),
        "x3": lambda n_samples, x1, x2: -0.5
        * (0.1 * x1 + 0.5 * (x2 - 0.0) ** 2 * n_samples),
    }
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def sanity_3_gen_old():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -3
        * (1 / (1 + np.exp(-1 * x1 ** 2 + n_samples)) - 0.4),
        # 'x3': lambda n_samples, x1, x2 : np.sin(x1) * np.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
        "x3": lambda n_samples, x1, x2: np.sin(x1) * np.exp(-((x1 + n_samples) ** 2))
        + x2 ** 2 * (n_samples - 2),
    }
    structural_equations_ts = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -3
        * (1 / (1 + torch.exp(-1 * x1 ** 2 + n_samples)) - 0.4),
        "x3": lambda n_samples, x1, x2: torch.sin(x1)
        * torch.exp(-((x1 + n_samples) ** 2))
        + x2 ** 2 * (n_samples - 2),
    }
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def sanity_3_gen_new():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        # 'x2': lambda n_samples, x1     :      - 2 * np.sign(n_samples) * (x1 * (1 + n_samples)) - 1,
        "x2": lambda n_samples, x1: np.sign(n_samples) * (x1 ** 2 + n_samples) / 5,
        "x3": lambda n_samples, x1, x2: -1 * np.sqrt(x1 ** 2 + x2 ** 2) + n_samples,
        # 'x3': lambda n_samples, x1, x2 : np.sin(x1) * np.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
    }
    structural_equations_ts = structural_equations_np
    # structural_equations_ts = {
    #   'x1': lambda n_samples,        :                                                             n_samples,
    #   'x2': lambda n_samples, x1     :           - 3 * (1 / (1 + torch.exp(- 1 * x1**2  + n_samples)) - 0.4),
    #   'x3': lambda n_samples, x1, x2 : torch.sin(x1) * torch.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
    # }
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
        "u2": Normal(0, 0.3),
        "u3": Normal(0, 1),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def sanity_3_gen():
    a0 = 0.25
    b = -1
    b0 = 0.1
    b1 = 1
    b2 = 1

    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: a0
        * np.sign(n_samples)
        * (x1 ** 2)
        * (1 + n_samples ** 2),
        "x3": lambda n_samples, x1, x2: b
        + b0 * np.sign(n_samples) * (b1 * x1 ** 2 + b2 * x2 ** 2)
        + n_samples,
    }
    structural_equations_ts = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: a0
        * torch.sign(torch.tensor(n_samples))
        * (x1 ** 2)
        * (1 + n_samples ** 2),
        "x3": lambda n_samples, x1, x2: b
        + b0 * torch.sign(torch.tensor(n_samples)) * (b1 * x1 ** 2 + b2 * x2 ** 2)
        + n_samples,
    }
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2.5, +2.5], [1, 1]),
        "u2": Normal(0, 0.5 ** 2),
        "u3": Normal(0, 0.25 ** 2),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def sanity_6_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples: n_samples,
        "x3": lambda n_samples: n_samples,
        "x4": lambda n_samples, x1, x2: x1 + 2 * x2 + n_samples,
        "x5": lambda n_samples, x2, x3, x4: x2 - x4 + 2 * x3 + n_samples,
        "x6": lambda n_samples, x1, x3, x4, x5: x3 + x4 - x5 + x1 + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
        "u2": MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
        "u3": MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
        "u4": Normal(0, 2),
        "u5": Normal(0, 2),
        "u6": Normal(0, 2),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def german_credit():
    e_0 = -1
    e_G = 0.5
    e_A = 1

    l_0 = 1
    l_A = 0.01
    l_G = 1

    d_0 = -1
    d_A = 0.1
    d_G = 2
    d_L = 1

    i_0 = -4
    i_A = 0.1
    i_G = 2
    # i_E = 10
    i_GE = 1

    s_0 = -4
    s_I = 1.5

    structural_equations_np = {
        # Gender
        "x1": lambda n_samples: n_samples,
        # Age
        "x2": lambda n_samples: -35 + n_samples,
        # Education
        "x3": lambda n_samples, x1, x2: -0.5
        + (
            1
            + np.exp(
                -(e_0 + e_G * x1 + e_A * (1 + np.exp(-0.1 * (x2))) ** (-1) + n_samples)
            )
        )
        ** (-1),
        # Loan amount
        "x4": lambda n_samples, x1, x2: l_0
        + l_A * (x2 - 5) * (5 - x2)
        + l_G * x1
        + n_samples,
        # Loan duration
        "x5": lambda n_samples, x1, x2, x4: d_0
        + d_A * x2
        + d_G * x1
        + d_L * x4
        + n_samples,
        # Income
        "x6": lambda n_samples, x1, x2, x3: i_0
        + i_A * (x2 + 35)
        + i_G * x1
        + i_GE * x1 * x3
        + n_samples,
        # Savings
        "x7": lambda n_samples, x6: s_0 + s_I * (x6 > 0) * x6 + n_samples,
    }
    structural_equations_ts = {
        # Gender
        "x1": lambda n_samples: n_samples,
        # Age
        "x2": lambda n_samples: -35 + n_samples,
        # Education
        "x3": lambda n_samples, x1, x2: -0.5
        + (
            1
            + torch.exp(
                -(
                    e_0
                    + e_G * x1
                    + e_A * (1 + torch.exp(-0.1 * (x2))) ** (-1)
                    + n_samples
                )
            )
        )
        ** (-1),
        # Loan amount
        "x4": lambda n_samples, x1, x2: l_0
        + l_A * (x2 - 5) * (5 - x2)
        + l_G * x1
        + n_samples,
        # Loan duration
        "x5": lambda n_samples, x1, x2, x4: d_0
        + d_A * x2
        + d_G * x1
        + d_L * x4
        + n_samples,
        # Income
        "x6": lambda n_samples, x1, x2, x3: i_0
        + i_A * (x2 + 35)
        + i_G * x1
        + i_GE * x1 * x3
        + n_samples,
        # Savings
        "x7": lambda n_samples, x6: s_0 + s_I * (x6 > 0) * x6 + n_samples,
    }
    noises_distributions = {
        # Gender
        "u1": Bernoulli(0.5),
        # Age
        "u2": Gamma(10, 3.5),
        # Education
        "u3": Normal(0, 0.5 ** 2),
        # Loan amount
        "u4": Normal(0, 2 ** 2),
        # Loan duration
        "u5": Normal(0, 3 ** 2),
        # Income
        "u6": Normal(0, 2 ** 2),
        # Savings
        "u7": Normal(0, 5 ** 2),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def adult():
    # TODO: change this placeholder
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,  # A_sex
        "x2": lambda n_samples: n_samples,  # C_age
        "x3": lambda n_samples: n_samples,  # C_nationality
        "x4": lambda n_samples, x1, x2, x3: n_samples,  # M_marital_status
        "x5": lambda n_samples, x1, x2, x3: n_samples,  # L_education_level / real-valued
        "x6": lambda n_samples, x1, x2, x3, x4, x5: n_samples,  # R_working_class
        "x7": lambda n_samples, x1, x2, x3, x4, x5: n_samples,  # R_occupation
        "x8": lambda n_samples, x1, x2, x3, x4, x5: n_samples,  # R_hours_per_week
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": Bernoulli(0.5, "-11"),
        "u2": Bernoulli(0.5, "-11"),
        "u3": Bernoulli(0.5, "-11"),
        "u4": Normal(0, 1),
        "u5": Normal(0, 1),
        "u6": Normal(0, 1),
        "u7": Normal(0, 1),
        "u8": Normal(0, 1),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def fair_imf_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,  # A
        "x2": lambda n_samples, x1: 0.5 * x1 + n_samples,  # A --> X_1
        "x3": lambda n_samples: n_samples,  # X_2
        "x4": lambda n_samples, x1: 0.5 * x1 + n_samples,  # A --> X_3
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": Bernoulli(0.5, "-11"),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions


def fair_cau_lin():
    # replaces previous 'fair-3-lin'
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,  # A
        "x2": lambda n_samples, x1: 0.5 * x1 + n_samples,  # A --> X_1
        "x3": lambda n_samples: n_samples,  # X_2
        "x4": lambda n_samples, x1, x2, x3: 0.5 * (x1 + x2 - x3)
        + n_samples,  # {X_1, X_2} --> X_3
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": Bernoulli(0.5, "-11"),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
    }

    return structural_equations_np, structural_equations_ts, noises_distributions


def fair_cau_anm():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,  # A
        "x2": lambda n_samples, x1: 0.5 * x1 + n_samples,  # A --> X_1
        "x3": lambda n_samples: n_samples,  # X_2
        "x4": lambda n_samples, x1, x2, x3: 0.5 * x1
        + 0.1 * (x2 ** 3 - x3 ** 3)
        + n_samples,  # {X_1, X_2} --> X_3
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": Bernoulli(0.5, "-11"),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
    }
    return structural_equations_np, structural_equations_ts, noises_distributions
