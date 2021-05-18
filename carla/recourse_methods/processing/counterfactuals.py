from typing import List

import numpy as np
import pandas as pd

from carla.models.api import MLModel


def counterfactual_to_dataframe(
    mlmodel: MLModel, counterfactuals: List
) -> pd.DataFrame:
    df_cfs = pd.DataFrame(
        np.array(counterfactuals), columns=mlmodel.feature_input_order
    )
    df_cfs[mlmodel.data.target] = np.argmax(mlmodel.predict_proba(df_cfs), axis=1)
    # Change all wrong counterfactuals to nan
    df_cfs.loc[df_cfs[mlmodel.data.target] == 0, :] = np.nan

    return df_cfs
