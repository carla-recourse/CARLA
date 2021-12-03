import numpy as np
import pandas as pd

from carla.data.causal_model import CausalModel


def get_noise_string(node):
    if not node[0] == "x":
        raise ValueError
    return "u" + node[1:]


def get_abduction_noise(
    node: str, parents, structural_equation, factual_instance: dict
):
    return factual_instance[node] - structural_equation(
        0, *[factual_instance[p] for p in parents]
    )


def sample_true_m0(
    node: str,
    scm: CausalModel,
    samples_df: pd.DataFrame,
    factual_instance: dict,
):
    # Step 1. [abduction]: compute noise or load from dataset using factual_instance
    # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
    # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
    parents = scm.get_parents(node)
    structural_equation = scm.structural_equations_np[node]

    predicted_noise = get_abduction_noise(
        node, parents, structural_equation, factual_instance
    )

    # # TODO why is this check here? Can it be moved to a test case
    # # TODO this can fail if data has no noise variables
    # true_noise = factual_instance("endogenous_and_exogenous")[f"u{node[1:]}"]
    # if (
    #     not scm.scm_class == "sanity-3-gen"
    # ):  # this scm uses multiplicative, rather then additive, noise
    #     if not np.abs(predicted_noise - true_noise) < 1e-5:
    #         raise ValueError("Noise{pred, true} are expected to be similiar")

    noise = np.array(predicted_noise)

    node_sample = structural_equation(noise, *[samples_df[p] for p in parents])
    return node_sample


def sample_true_m2(node: str, scm: CausalModel, samples_df: pd.DataFrame):
    # Step 1. [abduction]: compute noise or load from dataset using factual_instance
    # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
    # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
    parents = scm.get_parents(node)
    structural_equation = scm.structural_equations_np[node]

    noise = scm.noise_distributions[get_noise_string(node)].sample(samples_df.shape[0])

    node_sample = structural_equation(noise, *[samples_df[p] for p in parents])
    return node_sample
