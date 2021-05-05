from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

if __name__ == "__main__":

    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    feature_input_order_tf = [
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

    model = MLModelCatalog(data, "ann", feature_input_order_tf)
    print(f"Using model: {model.raw_model.__class__.__module__}")
    print(data.target)
    print(predict_negative_instances(model, data).head(100))

    feature_input_order_pt = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "sex_Female",
        "sex_Male",
        "workclass_Non-Private",
        "workclass_Private",
        "marital-status_Married",
        "marital-status_Non-Married",
        "occupation_Managerial-Specialist",
        "occupation_Other",
        "relationship_Husband",
        "relationship_Non-Husband",
        "race_Non-White",
        "race_White",
        "native-country_Non-US",
        "native-country_US",
    ]

    model_pt = MLModelCatalog(data, "ann", feature_input_order_pt, backend="pytorch")
    print(f"Using model: {model.raw_model.__class__.__module__}")
    print(predict_negative_instances(model_pt, data).head(100))
