import numpy as np

from carla.data.load_scm.distributions import (
    Bernoulli,
    Gamma,
    MixtureOfGaussians,
    Normal,
    Poisson,
    Uniform,
)


def test_normal():
    params = {"mean": 0, "var": 1}
    dist = Normal(params["mean"], params["var"])

    assert len(dist.sample(size=10)) == 10
    assert isinstance(dist.sample(size=1), float)

    # # for visual confirmation
    # dist.visualize()


def test_gaussian_mixture():
    params = {"probs": [0.9, 0.1], "means": [0, 10], "vars": [1, 1]}
    dist = MixtureOfGaussians(params["probs"], params["means"], params["vars"])

    assert len(dist.sample(size=10)) == 10
    assert isinstance(dist.sample(size=1), float)

    # # for visual confirmation
    # dist.visualize()


def test_uniform():
    params = {
        "lower": 0,
        "upper": 1,
    }
    dist = Uniform(params["lower"], params["upper"])

    s = dist.sample(size=10)
    assert len(s) == 10
    assert isinstance(dist.sample(size=1), float)
    assert np.all(s >= params["lower"]) and np.all(s <= params["upper"])

    # # for visual confirmation
    # dist.visualize()


def test_bernoulli():
    params = {
        "prob": 0.3,
    }
    dist = Bernoulli(params["prob"])

    s = dist.sample(size=10)
    assert len(s) == 10
    assert np.all([x in [0, 1] for x in s])

    # # for visual confirmation
    # dist.visualize()


def test_poisson():
    params = {
        "p_lambda": 4,
    }
    dist = Poisson(params["p_lambda"])

    s = dist.sample(size=10)

    assert len(s) == 10
    assert isinstance(dist.sample(size=1), (int, np.integer))
    assert np.all([x >= 0 for x in s])

    # # for visual confirmation
    # dist.visualize()


def test_gamma():
    params = {
        "shape": 2,
        "scale": 2,
    }
    dist = Gamma(params["shape"], params["scale"])

    s = dist.sample(size=10)

    assert len(s) == 10
    assert isinstance(dist.sample(size=1), float)
    assert np.all([x >= 0 for x in s])

    # # for visual confirmation
    # dist.visualize()
