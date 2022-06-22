import numpy as np

from carla.evaluation import distance


def test_l0():
    test_input_1 = np.array([[0, 1]])
    test_input_2 = np.array([[1, 0]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    delta = distance.get_delta(test_input_1, test_input_1)
    actual = distance.l0_distance(delta)
    expected = [0.0]

    assert actual == expected

    delta = distance.get_delta(test_input_1, test_input_2)
    actual = distance.l0_distance(delta)
    expected = [2.0]

    assert actual == expected

    expected = [1.0, 1.0]
    actual = distance.l0_distance(test_input_3)
    assert actual == expected


def test_d2():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -1]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distance.l1_distance(test_input_1)
    assert actual == expected

    expected = [2.0]
    actual = distance.l1_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 2.0]
    actual = distance.l1_distance(test_input_3)
    assert actual == expected


def test_d3():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -1]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distance.l2_distance(test_input_1)
    assert actual == expected

    expected = [2.0]
    actual = distance.l2_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 2.0]
    actual = distance.l2_distance(test_input_3)
    assert actual == expected


def test_d4():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -4]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distance.linf_distance(test_input_1)
    assert actual == expected

    expected = [4.0]
    actual = distance.linf_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 4.0]
    actual = distance.linf_distance(test_input_3)
    assert actual == expected


def test_distances():
    test_input_1 = np.array([[1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0]])
    test_input_2 = np.array([[1, 0, 0, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0]])

    expected = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    actual = distance.Distance().get_distances(test_input_1, test_input_2)
    assert actual == expected
