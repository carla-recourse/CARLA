import numpy as np

from carla.data.causal_model import CausalModel
from carla.recourse_methods.catalog.causal_recourse.samplers import get_abduction_noise


def test_abduction():

    scm = CausalModel("sanity-3-lin")
    data = scm.generate_dataset(100)

    factual_instance = data.df.iloc[0].to_dict()

    noise = data.noise.iloc[0].to_dict()

    # hard-coding the fact that we know variables have numbers as name and we have three of them
    for i in range(1, 4):

        true_noise = noise["u" + str(i)]

        parents = scm.get_parents("x" + str(i))
        predicted_noise = get_abduction_noise(
            "x" + str(i),
            parents,
            scm.structural_equations_np["x" + str(i)],
            factual_instance,
        )

        assert np.isclose(predicted_noise, true_noise, atol=0.1)
