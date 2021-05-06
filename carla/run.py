from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

if __name__ == "__main__":

    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    model = MLModelCatalog(data, "ann")
    print(f"Using model: {model.raw_model.__class__.__module__}")
    print(data.target)
    print(predict_negative_instances(model, data).head(100))

    model_pt = MLModelCatalog(data, "ann", backend="pytorch")
    print(f"Using model: {model.raw_model.__class__.__module__}")
    print(predict_negative_instances(model_pt, data).head(100))
