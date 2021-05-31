from typing import List

import yaml


def load_catalog(filename: str, dataset: str, keys: List[str]):
    with open(filename, "r") as f:
        catalog = yaml.safe_load(f)

    if dataset not in catalog:
        raise KeyError("Dataset not in catalog.")

    for key in keys:
        if key not in catalog[dataset].keys():
            raise KeyError("Important key {} is not in Catalog".format(key))
        if catalog[dataset][key] is None:
            catalog[dataset][key] = []

    return catalog[dataset]
