from abc import ABC

from carla.data.api import Data
from carla.data.catalog import DataCatalog
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.dice import Dice


def test_data():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog_yaml)

    assert issubclass(DataCatalog, Data)
    assert isinstance(data_catalog, Data)
    assert issubclass(Data, ABC)


def test_mlmodel():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog_yaml)

    model_catalog = MLModelCatalog(data, "ann")

    assert issubclass(MLModelCatalog, MLModel)
    assert isinstance(model_catalog, MLModel)
    assert issubclass(MLModel, ABC)


def test_cfmodel():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog_yaml)

    hyperparams = {"num": 1, "desired_class": 1}
    model_catalog = MLModelCatalog(data_catalog, "ann", use_pipeline=True)

    dice = Dice(model_catalog, hyperparams)

    assert issubclass(Dice, RecourseMethod)
    assert isinstance(dice, RecourseMethod)
    assert issubclass(RecourseMethod, ABC)
