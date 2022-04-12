from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm  # norm for univariate; use multivariate_normal otherwise
from scipy.stats import bernoulli


# univariate distributions
class BaseDistribution(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def sample(self, size: int):
        pass

    @abstractmethod
    def pdf(self, value):
        pass

    # TODO can abstract class have a filled in function?
    def visualize(self):
        plt.hist(self.sample(500), 50, facecolor="green", alpha=0.75)
        plt.ylabel("Count")
        plt.title(rf"Histogram of {self.name}")
        plt.grid(True)
        plt.show()


class Normal(BaseDistribution):
    def __init__(self, mean: Union[int, float], var: Union[int, float]):
        name = f"Normal(mean={mean}, var={var})"
        super().__init__(name)
        self.dist = norm(mean, np.sqrt(var))

    def sample(self, size=1) -> Union[np.ndarray, float]:
        s = self.dist.rvs(size)
        # returns a single number rather then a list if a single sample
        return s[0] if size == 1 else s

    def pdf(self, value):
        return self.dist.pdf(value)


class MixtureOfGaussians(BaseDistribution):
    def __init__(self, probs, means, vars):
        if not sum(probs) == 1:
            raise ValueError("Mixture probabilities must sum to 1.")
        if not len(probs) == len(means) == len(vars):
            raise ValueError("Length mismatch.")

        name = f"MoG(probs={probs}, means={means}, vars={vars})"
        super().__init__(name)

        self.probs = probs
        self.means = means
        self.vars = vars

    def sample(self, size=1) -> Union[np.ndarray, float]:
        mixtures = np.random.choice(len(self.probs), size=size, p=self.probs)
        s = [
            np.random.normal(self.means[mixture_idx], np.sqrt(self.vars[mixture_idx]))
            for mixture_idx in mixtures
        ]
        return s[0] if size == 1 else np.array(s)

    def pdf(self, value):
        return np.sum(
            [
                prob * norm(mean, var).pdf(value)
                for (prob, mean, var) in zip(self.probs, self.means, self.vars)
            ]
        )


class Uniform(BaseDistribution):
    def __init__(self, lower: Union[int, float], upper: Union[int, float]):
        if not lower < upper:
            raise ValueError("upper lower then lower")

        name = f"Uniform(lower={lower}, upper={upper})"
        super().__init__(name)

        self.lower = lower
        self.upper = upper

    def sample(self, size=1) -> Union[np.ndarray, float]:
        s = np.random.uniform(self.lower, self.upper, size=size)
        return s[0] if size == 1 else s

    def pdf(self, value):
        raise 1 / (self.upper - self.lower)


class Bernoulli(BaseDistribution):
    def __init__(self, prob: Union[int, float], btype="01"):
        if not 0 <= prob <= 1:
            raise ValueError("prob not in correct range")

        name = f"Bernoulli(prob={prob})"
        super().__init__(name)

        self.prob = prob
        self.btype = btype  # '01' is standard, '-11' also supported here

    def sample(self, size=1) -> Union[np.ndarray, int]:
        s = bernoulli.rvs(self.prob, size=size)
        if self.btype == "-11":
            s = s * 2 - 1
        return s[0] if size == 1 else s

    def pdf(self, value):
        raise Exception("not supported yet; code should not come here.")


class Poisson(BaseDistribution):
    def __init__(self, p_lambda: Union[int, float]):
        if p_lambda <= 0:
            raise ValueError("p_lambda should be strictly positive")

        name = f"Poisson(prob={p_lambda})"
        super().__init__(name)

        self.p_lambda = p_lambda

    def sample(self, size=1) -> Union[np.ndarray, np.integer]:
        s = np.random.poisson(self.p_lambda, size)
        return s[0] if size == 1 else s

    def pdf(self, value):
        raise Exception("not supported yet; code should not come here.")


class Gamma(BaseDistribution):
    def __init__(self, shape: Union[int, float], scale: Union[int, float]):
        if shape <= 0:
            raise ValueError("shape should be strictly positive")
        if scale <= 0:
            raise ValueError("scale should be strictly positive")

        name = f"Gamma(shape={shape}, scale={scale})"
        super().__init__(name)

        self.shape = shape
        self.scale = scale

    def sample(self, size=1) -> Union[np.ndarray, float]:
        s = np.random.gamma(self.shape, self.scale, size)
        return s[0] if size == 1 else s

    def pdf(self, value):
        raise Exception("not supported yet; code should not come here.")
