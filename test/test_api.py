from abc import ABC

import pytest

from carla.data.api import Data
from carla.data.catalog import DataCatalog
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.dice import Dice

testmodel = ["ann", "linear"]


def test_data():
    data_name = "adult"
    data_catalog = DataCatalog(data_name)

    assert issubclass(DataCatalog, Data)
    assert isinstance(data_catalog, Data)
    assert issubclass(Data, ABC)


@pytest.mark.parametrize("model_type", testmodel)
def test_mlmodel(model_type):
    data_name = "adult"
    data = DataCatalog(data_name)

    model_catalog = MLModelCatalog(data, model_type)

    assert issubclass(MLModelCatalog, MLModel)
    assert isinstance(model_catalog, MLModel)
    assert issubclass(MLModel, ABC)


def test_cfmodel():
    data_name = "adult"
    data_catalog = DataCatalog(data_name)

    hyperparams = {"num": 1, "desired_class": 1}
    model_catalog = MLModelCatalog(data_catalog, "ann", use_pipeline=True)

    dice = Dice(model_catalog, hyperparams)

    assert issubclass(Dice, RecourseMethod)
    assert isinstance(dice, RecourseMethod)
    assert issubclass(RecourseMethod, ABC)
