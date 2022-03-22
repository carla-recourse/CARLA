import numpy as np

from carla.data.catalog import OnlineCatalog
from carla.evaluation import redundancy
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


def test_redundancy():
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data.df)

    hyperparams = {"num": 1, "desired_class": 1}
    test_factual = factuals.iloc[:5]

    cfs = Dice(model_tf, hyperparams).get_counterfactuals(factuals=test_factual)

    red = redundancy(factuals, cfs, model_tf)

    expected = (5, 1)
    actual = np.array(red).shape

    assert expected == actual
