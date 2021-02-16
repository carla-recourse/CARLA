from carla import distances


def test_d1():
    actual = distances.d1_distance([0, 1], [0, 1])
    expected = 0

    assert actual == expected


def test_d4():
    actual = distances.d4_distance([0, 1], [0, 1])
    expected = 0

    assert actual == expected


def test_counterfactuals():

    factual = [
        23,
        "Private",
        134446,
        "HS-grade",
        9,
        "Separated",
        "Machine-op-inpct",
        "Unmarried",
        "Black",
        "Male",
        0,
        2356,
        1,
        "United States",
    ]

    counterfactual = [
        40,
        "Without-Pay",
        132969,
        "HS-grade",
        9,
        "Divorced",
        "Machine-op-inpct",
        "Unmarried",
        "Black",
        "Male",
        98982,
        3556,
        1,
        "United States",
    ]

    actual = distances.d1_distance(factual, counterfactual)
    expected = 6
    assert actual == expected

    actual = distances.d4_distance(factual, counterfactual)
    expected = 98982
    assert actual == expected
