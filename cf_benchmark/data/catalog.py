import yaml

from .load_data import load_dataset

CATALOG_FILE = "data_catalog.yaml"


class DataCatalog:
    def __init__(self, dataset):
        self.data = load_dataset(dataset)
        self.catalog = self.load_catalog(CATALOG_FILE, dataset)

    def load_catalog(self, filename, dataset):
        with open(filename, "r") as f:
            catalog = yaml.safe_load(f)

        if dataset not in catalog:
            raise KeyError("Dataset not in catalog.")

        return catalog[dataset]

    @property
    def categoricals(self):
        return self.catalog["categorical"]

    @property
    def continous(self):
        return self.catalog["continous"]

    @property
    def immutables(self):
        return self.catalog["immutable"]

    @property
    def target(self):
        return self.catalog["target"]
