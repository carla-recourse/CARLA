from carla.data import DataCatalog
from carla.models import load_model


def predict_label(model, data, label):

    print(model)
    print(data)
    pass


if __name__ == "__main__":

    data_name = "adult"
    data_catalog = "data_catalog.yaml"
    data = DataCatalog(data_name)

    print(data.categoricals)
    print(data.continous)
    print(data.immutables)
    print(data.target)
    print(data.normalized)
    print(data.encoded)
    print(data.encoded.dtypes)

    model = load_model("ann", data_name)
