from carla.cf_models.catalog.dice import Dice
from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances


def test_dice_get_counterfactuals():
    # Build data and mlmodel
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)
    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:22]

    cfs = Dice(model_tf, data).get_counterfactuals(
        factuals=test_factual, num_of_cf=1, desired_class=1
    )

    assert test_factual.shape[0] == cfs.shape[0]
