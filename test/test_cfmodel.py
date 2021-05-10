from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.clue import Clue
from carla.recourse_methods.catalog.dice import Dice


def test_dice_get_counterfactuals():
    # Build data and mlmodel
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

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

    model_tf = MLModelCatalog(data, "ann", feature_input_order)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    cfs = Dice(model_tf, hyperparams).get_counterfactuals(factuals=test_factual)

    assert test_factual.shape[0] == cfs.shape[0]


def test_clue():
    # Build data and mlmodel
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "sex_Female",
        "sex_Male",
        "workclass_Non-Private",
        "workclass_Private",
        "marital-status_Married",
        "marital-status_Non-Married",
        "occupation_Managerial-Specialist",
        "occupation_Other",
        "relationship_Husband",
        "relationship_Non-Husband",
        "race_Non-White",
        "race_White",
        "native-country_Non-US",
        "native-country_US",
    ]

    model = MLModelCatalog(data, "ann", feature_input_order, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:5]

    hyperparams = {
        "data_name": "adult",
        "train_vae": False,
        "width": 10,
        "depth": 3,
        "latent_dim": 12,
    }
    cfs = Clue(data, model, hyperparams).get_counterfactuals(test_factual)
    print(cfs)
