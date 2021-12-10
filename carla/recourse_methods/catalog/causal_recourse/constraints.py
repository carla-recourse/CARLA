from .sampler import Sampler


def point_constraint(scm, factual_instance, action_set, sampling_handle, mlmodel):

    # if action set is empty, return false as we don't flip the label with a factual instance
    if not bool(action_set):
        return False

    sampler = Sampler(scm)
    cf_instance = sampler.sample(1, factual_instance, action_set, sampling_handle)

    # TODO crop samples to fall within [0, 1] for MinMaxScaler
    prediction = mlmodel.predict(cf_instance)
    return prediction.round() == 1
