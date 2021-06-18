from carla.recourse_methods.processing.counterfactuals import check_hyperparams


def test_check_hyperparams():
    hyperparams = {"key1": 1, "key3": "3", "key5": {"sub_key1": 22}}

    default = {
        "key1": 22,
        "key2": "_optional_",
        "key3": "1",
        "key4": {"sub_key1": 1, "sub_key2": 2},
        "key5": {"sub_key1": None},
    }

    actual = check_hyperparams(hyperparams, default)

    expected = {
        "key1": 1,
        "key2": None,
        "key3": "3",
        "key4": {"sub_key1": 1, "sub_key2": 2},
        "key5": {"sub_key1": 22},
    }

    assert actual == expected
