import yaml


def load_catalog(filename, dataset):
    with open(filename, "r") as f:
        catalog = yaml.safe_load(f)

    if dataset not in catalog:
        raise KeyError("Dataset not in catalog.")

    return catalog[dataset]
