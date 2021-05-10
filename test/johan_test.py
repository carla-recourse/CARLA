# flake8: noqa
import pandas as pd
import tensorflow as tf
from keras import activations
from keras.layers import Activation, Dense
from keras.models import Model, Sequential, model_from_json
from tensorflow import Graph, Session

from carla.data.catalog.catalog import DataCatalog
from carla.models.catalog.catalog import MLModelCatalog

# from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.cem.cem import CEM


def load_AE():

    model_filename = "ae_tf.json"
    weigth_filename = "ae_tf.h5"

    json_file = open(model_filename, "r")
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()
    decoder.load_weights(weigth_filename)

    return decoder


class Model_Tabular:
    def __init__(
        self,
        dim_input,
        dim_hidden_layer1,
        dim_hidden_layer2,
        dim_output_layer,
        num_of_classes,
        restore=None,
        use_prob=False,
    ):

        # For model loading
        """
        :param dim_input: int > 0, number of neurons for this layer
        :param dim_hidden_layer_1: int > 0, number of neurons for this layer
        :param dim_hidden_layer_2: int > 0, number of neurons for this layer
        :param dim_output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        :param use_prob: boolean; FALSE required for CEM; all others should use True
        """  #

        self.dim_input = dim_input
        self.dim_hidden_layer1 = dim_hidden_layer1
        self.dim_hidden_layer2 = dim_hidden_layer2
        self.dim_output_layer = dim_output_layer
        self.num_of_classes = num_of_classes

        model = Sequential(
            [
                Dense(
                    self.dim_hidden_layer1, input_dim=self.dim_input, activation="relu"
                ),
                Dense(self.dim_hidden_layer2, activation="relu"),
                Dense(self.dim_output_layer, activation="relu"),
                Dense(self.num_of_classes),
            ]
        )

        # whether to output probability
        if use_prob:
            model.add(Activation(activations.softmax))
        if restore:
            model.load_weights(restore)
            # model.summary()

        self.model = model

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        return self.model(data)


data = DataCatalog(data_name="adult", catalog_file="adult_catalog.yaml")


feature_input_order = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "hours-per-week",
    "capital-loss",
    "workclass_Private",
    "marital-status_Non-Married",
    "occupation_Other",
    "relationship_Non-Husband",
    "race_White",
    "sex_Male",
    "native-country_US",
]

# model_ae = MLModelCatalog(
#     data=data, model_type="ae", feature_input_order=feature_input_order
# )

# factuals = predict_negative_instances(model_ann, data)
factuals = pd.read_csv("factuals.csv")

hyperparams = {"num": 1, "desired_class": 1}

hyperparams_cem = {"kappa": 0.1, "beta": 0.9, "gamma": 0.0, "mode": "PN"}
test_factuals = factuals.iloc[:5]

# recourse = Dice(mlmodel=model, data=data, hyperparams=hyperparams)
# recourse.get_counterfactuals(factuals=test_factual)

# model_ann.use_pipeline = True

# Load TF ANN
graph = Graph()
with graph.as_default():
    ann_sess = Session()
    with ann_sess.as_default():
        model_ann = MLModelCatalog(
            data=data, model_type="ann", feature_input_order=feature_input_order
        )
        # model_path_tf_13 = "ann_tf_adult_full_input_13"
        model_path_tf_13 = "ann.h5"
        ann_tf_13 = Model_Tabular(
            13, 18, 9, 3, 2, restore=model_path_tf_13, use_prob=True
        )
        model_ae = load_AE()

        foo = test_factuals.drop(["income"], axis=1)
        foo = model_ann.perform_pipeline(foo)
        # result1 = model_ann.predict(foo)
        result1 = model_ann.raw_model.predict(foo)
        result2 = ann_tf_13.model.predict(foo)


with graph.as_default():
    with ann_sess.as_default():
        foo = model_ann.raw_model
        recourse = CEM(
            sess=ann_sess,
            catalog_model=model_ann,
            mode=hyperparams_cem["mode"],
            AE=model_ae,
            batch_size=1,
            kappa=hyperparams_cem["kappa"],
            init_learning_rate=1e-2,
            binary_search_steps=9,
            max_iterations=100,
            initial_const=10,
            beta=hyperparams_cem["beta"],
            gamma=hyperparams_cem["gamma"],
        )
        result = recourse.get_counterfactuals(factuals=test_factuals)
        instance_list, cf_list, times_list, succes_rate = result
