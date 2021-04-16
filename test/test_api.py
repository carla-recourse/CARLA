from abc import ABC

from carla.data.api import Data
from carla.data.catalog import DataCatalog


def test_data():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog)

    assert issubclass(DataCatalog, Data)
    assert isinstance(data_catalog, Data)
    assert issubclass(Data, ABC)
