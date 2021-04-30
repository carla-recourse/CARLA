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
    data_catalog = DataCatalog(data_name, data_catalog_yaml, drop_first_encoding=True)

    assert issubclass(DataCatalog, Data)
    assert isinstance(data_catalog, Data)
    assert issubclass(Data, ABC)


def test_mlmodel():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog_yaml, drop_first_encoding=True)

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
    model_catalog = MLModelCatalog(data, "ann", feature_input_order)

    assert issubclass(MLModelCatalog, MLModel)
    assert isinstance(model_catalog, MLModel)
    assert issubclass(MLModel, ABC)


def test_cfmodel():
    data_name = "adult"
    data_catalog_yaml = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog_yaml, drop_first_encoding=True)

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
    hyperparams = {"num": 1, "desired_class": 1}
    model_catalog = MLModelCatalog(
        data_catalog, "ann", feature_input_order, use_pipeline=True
    )

    dice = Dice(model_catalog, data_catalog, hyperparams)

    assert issubclass(Dice, RecourseMethod)
    assert isinstance(dice, RecourseMethod)
    assert issubclass(RecourseMethod, ABC)
