from abc import ABC

from carla.cf_models.api import CFModel
from carla.cf_models.catalog.dice import Dice
from carla.data.api import Data
from carla.data.catalog import DataCatalog
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog


def test_data():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog_yaml, True)

    assert issubclass(DataCatalog, Data)
    assert isinstance(data_catalog, Data)
    assert issubclass(Data, ABC)


def test_mlmodel():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog_yaml, True)
    model_catalog = MLModelCatalog(data_catalog, data_name, "ann")

    assert issubclass(MLModelCatalog, MLModel)
    assert isinstance(model_catalog, MLModel)
    assert issubclass(MLModel, ABC)


def test_cfmodel():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog_yaml, True)
    model_catalog = MLModelCatalog(data_catalog, data_name, "ann")

    dice = Dice(model_catalog, data_catalog)

    assert issubclass(Dice, CFModel)
    assert isinstance(dice, CFModel)
    assert issubclass(CFModel, ABC)
