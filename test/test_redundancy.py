import numpy as np

from carla.data.catalog import DataCatalog
from carla.evaluation import redundancy
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.models.pipelining import encode, scale
from carla.recourse_methods.catalog.dice import Dice


def test_redundancy():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    cfs = Dice(model_tf, hyperparams).get_counterfactuals(factuals=test_factual)
    model_tf.use_pipeline = False

    df_enc_norm_fact = scale(model_tf.scaler, data.continous, factuals)
    df_enc_norm_fact = encode(model_tf.encoder, data.categoricals, df_enc_norm_fact)
    df_enc_norm_fact = df_enc_norm_fact[model_tf.feature_input_order]

    red = redundancy(df_enc_norm_fact, cfs, model_tf)

    expected = (5, 1)
    actual = np.array(red).shape

    assert expected == actual
