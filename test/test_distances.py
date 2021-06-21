import numpy as np

from carla.evaluation import distances


def test_d1():
    test_input_1 = np.array([[0, 1]])
    test_input_2 = np.array([[1, 0]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    delta = distances.get_delta(test_input_1, test_input_1)
    actual = distances.d1_distance(delta)
    expected = [0.0]

    assert actual == expected

    delta = distances.get_delta(test_input_1, test_input_2)
    actual = distances.d1_distance(delta)
    expected = [2.0]

    assert actual == expected

    expected = [1.0, 1.0]
    actual = distances.d1_distance(test_input_3)
    assert actual == expected


def test_d2():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -1]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distances.d2_distance(test_input_1)
    assert actual == expected

    expected = [2.0]
    actual = distances.d2_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 2.0]
    actual = distances.d2_distance(test_input_3)
    assert actual == expected


def test_d3():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -1]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distances.d3_distance(test_input_1)
    assert actual == expected

    expected = [2.0]
    actual = distances.d3_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 2.0]
    actual = distances.d3_distance(test_input_3)
    assert actual == expected


def test_d4():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -4]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distances.d4_distance(test_input_1)
    assert actual == expected

    expected = [4.0]
    actual = distances.d4_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 4.0]
    actual = distances.d4_distance(test_input_3)
    assert actual == expected


def test_distances():
    test_input_1 = np.array([[1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0]])
    test_input_2 = np.array([[1, 0, 0, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0]])

    expected = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    actual = distances.get_distances(test_input_1, test_input_2)
    assert actual == expected
