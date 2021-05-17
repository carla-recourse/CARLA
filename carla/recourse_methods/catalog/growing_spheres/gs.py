import pandas as pd

from carla.recourse_methods.api import RecourseMethod


class GrowingSpheres(RecourseMethod):
    def __init__(self, mlmodel):
        super().__init__(mlmodel)

    def get_counterfactuals(self, factuals: pd.DataFrame):
        pass


'''
def get_counterfactual(data, df_instances, model):
    """
    :param dataset_path: str; path
    :param df_instances: pd data frame; df_instances to generate counterfactuals for
    :param model; classification model (either tf keras, pytorch or sklearn)
    :return: input df_instances & counterfactual explanations
    """

    # 1. Drop target of data and df_instances
    # 2. Normalize instance by data?
    # 3. Robust binarization?
    # 4. Choose binary, continous and immutable cols
    # 5. For all df_instances use `growing_spheres_search` to get counterfactual candidate
    # 6. Track time (persist somewhere?)
    # 7. Track non-successfull counterfactual finds
    # ... predict labels?
    # ... where is checked that its actually a counterfactual?
    # ... convert back into non-binarized, non-scaled counterfactual?

    df_instances = df_instances.drop(data.target, axis="columns")
    instance = df_instances.iloc[0]
    start = timeit.default_timer()
    counterfactuals = []
    times_list = []

    mutables = list(set(data.raw.columns) - set(data.immutables) - set([data.target]))

    counterfactual = gs.growing_spheres_search(
        instance, mutables, data.immutables, data.continous, data.categoricals, model
    )
    stop = timeit.default_timer()
    time_taken = stop - start

    # counterfactuals.append(counterfactual)
    # times_list.append(time_taken)

    # counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
    # counterfactuals_df.columns = df_instances.columns

    # # Success rate & drop not successful counterfactuals & process remainder
    # success_rate, counterfactuals_indeces = measure.success_rate_and_indices(
    #     counterfactuals_df
    # )
    # counterfactuals_df = counterfactuals_df.iloc[counterfactuals_indeces]
    # df_instances = df_instances.iloc[counterfactuals_indeces]

    # # Obtain labels
    # instance_label = np.argmax(model.model.predict(df_instances.values), axis=1)
    # counterfactual_label = np.argmax(
    #     model.model.predict(counterfactuals_df.values), axis=1
    # )

    # # Round binary columns to integer
    # counterfactuals_df[binary_cols] = (
    #     counterfactuals_df[binary_cols].round(0).astype(int)
    # )

    # # Order counterfactuals and df_instances in original data order
    # counterfactuals_df = counterfactuals_df[data.columns]
    # df_instances = df_instances[data.columns]

    # if len(binary_cols) > 0:

    #     # Convert binary cols of counterfactuals and df_instances into strings: Required for >>Measurement<< in script
    #     counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].astype(
    #         "string"
    #     )
    #     df_instances[binary_cols] = df_instances[binary_cols].astype("string")

    #     # Convert binary cols back to original string encoding
    #     counterfactuals_df = preprocessing.map_binary_backto_string(
    #         data, counterfactuals_df, binary_cols
    #     )
    #     df_instances = preprocessing.map_binary_backto_string(data, df_instances, binary_cols)

    # # Add labels
    # counterfactuals_df[target_name] = counterfactual_label
    # df_instances[target_name] = instance_label

    # # Collect in list making use of pandas
    # instances_list = []
    # counterfactuals_list = []
    # for i in range(counterfactuals_df.shape[0]):
    #     counterfactuals_list.append(
    #         pd.DataFrame(
    #             counterfactuals_df.iloc[i].values.reshape((1, -1)),
    #             columns=counterfactuals_df.columns,
    #         )
    #     )
    #     instances_list.append(
    #         pd.DataFrame(
    #             df_instances.iloc[i].values.reshape((1, -1)), columns=df_instances.columns
    #         )
    #     )

    # return instances_list, counterfactuals_list, times_list, success_rate

    return None
'''
