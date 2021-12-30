from .sampler import Sampler


def point_constraint(scm, factual_instance, action_set, sampling_handle, mlmodel):
    """
    Check if perturbed factual instance is a counterfactual.

    Parameters
    ----------
    scm: StructuralCausalModel
        Needed to create new samples.
    factual_instance: pd.Series
        Contains a single factual instance, where each element corresponds to a feature.
    action_set: dict
        Contains perturbation of features.
    sampling_handle: function
        Function that control the sampling.
    mlmodel: MLModelCatalog
        The classifier.

    Returns
    -------
    bool
    """

    # if action set is empty, return false as we don't flip the label with a factual instance
    if not bool(action_set):
        return False

    sampler = Sampler(scm)
    cf_instance = sampler.sample(1, factual_instance, action_set, sampling_handle)

    prediction = mlmodel.predict(cf_instance)
    return prediction.round() == 1
