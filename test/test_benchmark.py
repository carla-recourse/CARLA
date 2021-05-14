from carla.data.catalog import DataCatalog
from carla.evaluation import Benchmark
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


def test_benchmarks():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
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


def test_distances():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    dict_distances = benchmark.compute_distances()

    expected = 5
    actual = len(dict_distances["Distances"])
    assert expected == actual
