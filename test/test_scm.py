import numpy as np

from carla.data.causal_model import CausalModel


def test_load_scm():
    def remove_prefix(node):
        """replaces e.g. x101 or u101 with 101"""
        assert node[0] == "x" or node[0] == "u"
        return node[1:]

    scm = CausalModel("sanity-3-lin")

    # keys should have the same name
    assert (
        [remove_prefix(node) for node in scm.structural_equations_np.keys()]
        == [remove_prefix(node) for node in scm.structural_equations_ts.keys()]
        == [remove_prefix(node) for node in scm.noise_distributions.keys()]
    )

    # endogenous variables must start with x
    assert np.all(["x" in node for node in scm.structural_equations_np.keys()])
    assert np.all(["x" in node for node in scm.structural_equations_ts.keys()])


def test_synthetic_data():

    scm = CausalModel("sanity-3-lin")

    dataset = scm.generate_dataset(10)

    assert dataset.df.shape == (10, 4)
    assert dataset.noise.shape == (10, 3)
    assert set(dataset.continuous) == {"x1", "x2", "x3"}
    assert dataset.categorical == []
    assert dataset.target == "label"
