import dice_ml
import pandas as pd

from ...api import RecourseMethod


class Dice(RecourseMethod):
    def __init__(self, mlmodel, data, hyperparams):
        """
        Constructor for Dice model
        Implementation can be seen at https://github.com/interpretml/DiCE

        Restrictions:
        ------------
        -   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        -   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

        Parameters
        ----------
        mlmodel : models.api.MLModel
            ML model to build counterfactuals for.
        data : data.api.Data
            Underlying dataset we want to build counterfactuals for.
        hyperparams : dict
            Hyperparameter which are needed for DICE to generate counterfactuals.
            Structure: {"num": int, "desired_class": int}
        """
        # Prepare data for dice data structure
        self._dice_data = dice_ml.Data(
            dataframe=data.raw,
            continuous_features=data.continous,
            outcome_name=data.target,
        )

        self._dice_model = dice_ml.Model(model=mlmodel, backend="sklearn")

        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="random")
        self._num = hyperparams["num"]
        self._desired_class = hyperparams["desired_class"]

    @property
    def dice_model(self):
        return self._dice

    def get_counterfactuals(self, factuals):
        """
        Compute a certain number of counterfactuals per factual example.


        Parameters
        ----------
        factuals : pd.DataFrame
            DataFrame containing all samples for which we want to generate counterfactual examples.
            All instances should belong to the same class.
        num : int
            Number of counterfactuals we want to generate per factual
        desired_class : int
            The target class we want to reach for our factuals

        Returns
        -------

        """

        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        # Generate counterfactuals
        dice_exp = self._dice.generate_counterfactuals(
            querry_instances, total_CFs=self._num, desired_class=self._desired_class
        )

        cf_ex_list = dice_exp.cf_examples_list
        cf = pd.concat([cf_ex.final_cfs_df for cf_ex in cf_ex_list], ignore_index=True)

        # TODO: Expandable for further functionality

        return cf
