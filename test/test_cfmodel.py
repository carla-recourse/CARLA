from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


def test_dice_get_counterfactuals():
    # Build data and mlmodel
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]
    encoding = [
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    model_tf = MLModelCatalog(data, "ann", feature_input_order, encoding)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:22]

    hyperparams = {"num": 1, "desired_class": 1}

    cfs = Dice(model_tf, data, hyperparams).get_counterfactuals(factuals=test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
