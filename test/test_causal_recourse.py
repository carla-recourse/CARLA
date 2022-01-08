from carla.data.causal_model import CausalModel
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)


def test_causal_recourse():

    scm = CausalModel("sanity-3-lin")
    data = scm.generate_dataset(10000)

    training_params = {"lr": 0.8, "epochs": 10, "batch_size": 16}

    model_type = "linear"
    model = MLModelCatalog(data, model_type, load_online=False, use_pipeline=False)
    model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
    )

    # get factuals
    factuals = predict_negative_instances(model, data, normalize=False)[:5]
    assert len(factuals) > 0

    hyperparams = {
        "optimization_approach": "brute_force",
        "num_samples": 10,
        "scm": scm,
        "constraint_handle": constraints.point_constraint,
        "sampler_handle": samplers.sample_true_m0,
    }
    cfs = CausalRecourse(model, hyperparams).get_counterfactuals(factuals)

    # cfs = model.perform_inverse_pipeline(cfs)

    assert len(cfs) == len(factuals)
