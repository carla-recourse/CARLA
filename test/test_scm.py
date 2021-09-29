import numpy as np

from carla.data.load_scm import load_scm


def test_load_scm():
    def _remove_prefix(node):
        """replaces e.g. x101 or u101 with 101"""
        assert node[0] == "x" or node[0] == "u"
        return node[1:]

    scm = load_scm("sanity-3-lin")

    # keys should have the same name
    assert (
        [_remove_prefix(node) for node in scm.structural_equations_np.keys()]
        == [_remove_prefix(node) for node in scm.structural_equations_ts.keys()]
        == [_remove_prefix(node) for node in scm.noise_distributions.keys()]
    )

    # endogenous variables must start with x
    assert np.all(["x" in node for node in scm.structural_equations_np.keys()])
    assert np.all(["x" in node for node in scm.structural_equations_ts.keys()])
