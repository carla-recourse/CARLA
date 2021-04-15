from carla.data import DataCatalog
from carla.methods import gs
from carla.models import load_model, predict_negative_instances

if __name__ == "__main__":

    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    model = load_model("ann", data_name)
    print(f"Using model: {model.__class__.__module__}")
    print(data.target)

    instances = predict_negative_instances(model, data).head(10)

    cf = gs.get_counterfactual(data, instances, model)
