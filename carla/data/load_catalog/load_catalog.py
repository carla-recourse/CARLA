import os
from typing import List

import yaml

from carla import lib_path


def load_catalog(filename: str, dataset: str, keys: List[str]):
    with open(os.path.join(lib_path, filename), "r") as f:
        catalog = yaml.safe_load(f)

    if dataset not in catalog:
        raise KeyError(f"Dataset '{dataset}' not in catalog.")

    # TODO: Use schema validation instead of passing required keys
    for key in keys:
        if key not in catalog[dataset].keys():
            raise KeyError(f"Required key {key} is not in Catalog")
        if catalog[dataset][key] is None:
            catalog[dataset][key] = []

    return catalog[dataset]
