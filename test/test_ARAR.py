import numpy as np
import pandas as pd
import pytest

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.arar import ARAR


@pytest.mark.parametrize("model_type", ["ann", "linear"])
def test_arar(model_type):
    """Tests that the provided output is in the expected format"""
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:10]
    hyperparams = {"outer_iters": 2}
    df_cfs = ARAR(model, hyperparams).get_counterfactuals(test_factual)
    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", ["ann", "linear"])
def test_arar_flipped_label(model_type):
    """Tests that the prediction of the generated counterfactual examples are flipped"""
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:10]
    hyperparams = {"outer_iters": 2, "y_target": 1}
    df_cfs = ARAR(model, hyperparams).get_counterfactuals(test_factual).dropna()
    labels = model.predict(df_cfs)
    res = np.all(labels >= 0.5)
    assert res
    hyperparams = {"outer_iters": 2, "y_target": 0}
    factuals = predict_negative_instances(model, data.df, negative_class=1)
    test_factual = factuals.iloc[:10]
    df_cfs = ARAR(model, hyperparams).get_counterfactuals(test_factual).dropna()
    labels = model.predict(df_cfs)
    res = np.all(labels <= 0.5)
    assert res
