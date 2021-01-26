from cf_benchmark.data import load_dataset
from cf_benchmark.models import load_model


def predict_label(model, data, label):

    print(model)
    print(data)
    pass


if __name__ == "__main__":

    data_name = "adult"
    data = load_dataset(data_name)
    model = load_model("ann", data_name)
    target = "income"

    predict_label(model, data, target)
