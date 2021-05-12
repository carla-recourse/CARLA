from carla.data.catalog import DataCatalog
from carla.evaluation import Benchmark
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


def test_benchmarks():
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

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    dict_benchmark = benchmark.run_benchmark()

    expected = ["Distances"]
    actual = list(dict_benchmark.keys())

    assert expected == actual

    test = {"test": {"test 1": 1, "test 2": 2, "test 3": 3}}
    benchmark.to_csv({**dict_benchmark, **test}, "test.csv")


def test_distances():
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

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    dict_distances = benchmark.compute_distances()

    expected = ["Distance 1", "Distance 2", "Distance 3", "Distance 4"]
    actual = list(dict_distances["Distances"].keys())
    assert expected == actual
