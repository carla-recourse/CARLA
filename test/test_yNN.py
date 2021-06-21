from carla.data.catalog import DataCatalog
from carla.evaluation import yNN
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


def test_yNN():
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
    cfs = dice.get_counterfactuals(factuals=test_factual)

    model_tf.use_pipeline = False
    ynn = yNN(cfs, dice, model_tf, 5)

    assert 0 <= ynn <= 1
