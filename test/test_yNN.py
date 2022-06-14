from carla.data.catalog import OnlineCatalog
from carla.evaluation import yNN
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice


def test_yNN():
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann", backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model_tf, data.df)

    hyperparams = {"num": 1, "desired_class": 1}
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)
    cfs = dice.get_counterfactuals(factuals=test_factual)

    ynn = yNN(cfs, model_tf, 5, cf_label=1)

    assert 0 <= ynn <= 1
