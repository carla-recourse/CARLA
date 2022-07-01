from functools import lru_cache

import carla.evaluation.catalog as evaluation_catalog
from carla.data.catalog import OnlineCatalog
from carla.evaluation import Benchmark
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


@lru_cache(maxsize=None)
def make_benchmark(data_name="adult", model_name="ann"):
    # get data and mlmodel
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, model_name, backend="tensorflow")

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    # get recourse method
    hyperparams = {"num": 1, "desired_class": 1}
    recourse_method = Dice(model, hyperparams)

    # make benchmark object
    benchmark = Benchmark(model, recourse_method, test_factual)

    return benchmark


@lru_cache(maxsize=None)
def run_benchmark():
    benchmark = make_benchmark()
    evaluation_measures = [
        evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
        evaluation_catalog.Distance(benchmark.mlmodel),
        evaluation_catalog.SuccessRate(),
        evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
        evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
        evaluation_catalog.AvgTime({"time": benchmark.timer}),
    ]
    df_benchmark = benchmark.run_benchmark(evaluation_measures)
    return df_benchmark, evaluation_measures


def test_benchmarks():
    # Build data and mlmodel
    benchmark, _ = run_benchmark()

    expected = (5, 9)
    actual = benchmark.shape
    assert expected == actual


def test_ynn():
    benchmark, evaluation_measures = run_benchmark()

    ynn = evaluation_measures[0]
    ynn_benchmark = benchmark[ynn.columns].dropna()

    expected = (1, 1)
    actual = ynn_benchmark.shape
    assert expected == actual
    assert 0 <= ynn_benchmark.values <= 1


def test_distances():
    benchmark, evaluation_measures = run_benchmark()
    distance = evaluation_measures[1]
    distance_benchmark = benchmark[distance.columns]

    expected = (5, 4)
    actual = distance_benchmark.shape
    assert expected == actual


def test_success_rate():
    benchmark, evaluation_measures = run_benchmark()
    success_rate = evaluation_measures[2]
    sr_benchmark = benchmark[success_rate.columns].dropna()

    expected = (1, 1)
    actual = sr_benchmark.shape
    assert expected == actual


def test_redundancy():
    benchmark, evaluation_measures = run_benchmark()
    redundancy = evaluation_measures[3]
    redundancy_benchmark = benchmark[redundancy.columns]

    expected = (5, 1)
    actual = redundancy_benchmark.shape
    assert expected == actual


def test_violation():
    benchmark, evaluation_measures = run_benchmark()
    constraint_violation = evaluation_measures[4]
    violation_benchmark = benchmark[constraint_violation.columns]

    expected = (5, 1)
    actual = violation_benchmark.shape
    assert expected == actual


def test_time():
    # Build data and mlmodel
    benchmark, evaluation_measures = run_benchmark()
    time_measure = evaluation_measures[5]
    time_benchmark = benchmark[time_measure.columns].dropna()

    expected = (1, 1)
    actual = time_benchmark.shape
    assert expected == actual
