import pandas as pd
import torch
from recourse_fare.environment import Environment


class MockEnv(Environment):
    def __init__(
        self, f, model, preprocessor, preprocessor_fare, programs_library, arguments
    ):

        self.preprocessor = preprocessor
        self.preprocessor_fare = preprocessor_fare

        self.prog_to_func = None  # We don't need this since we hacked the act method
        self.prog_to_precondition = (
            None  # We don't need this since we hacked the "can_be_called" method
        )

        self.prog_to_postcondition = self._intervene_postcondition

        self.programs_library = programs_library
        self.arguments = arguments

        self.max_depth_dict = 12

        # Call parent constructor
        super().__init__(
            f,
            model,
            self.prog_to_func,
            self.prog_to_precondition,
            self.prog_to_postcondition,
            self.programs_library,
            self.arguments,
            self.max_depth_dict,
        )

    def can_be_called(self, program_index, args_index):
        program = self.get_program_from_index(program_index)
        args = self.complete_arguments[args_index]

        mask_over_args = self.get_mask_over_args(program_index)
        if mask_over_args[args_index] == 0:
            return False

        if program == "STOP":
            return True

        if self.features[program] == args:
            return False

        if isinstance(args, int) or isinstance(args, float):
            return self.features[program] + args >= 0

        return True

    def act(self, primary_action, arguments=None):
        assert self.has_been_reset, "Need to reset the environment before acting"
        assert (
            primary_action in self.primary_actions
        ), "action {} is not defined".format(primary_action)

        # The primary action is basically the feature we want to change.
        # Therefore, based on the argument type, we change it accordingly

        if primary_action == "STOP":
            return self.get_observation()

        if isinstance(arguments, int) or isinstance(arguments, float):
            self.features[primary_action] += arguments
        else:
            self.features[primary_action] = arguments

        return self.get_observation()

    def get_state_str(self, state):
        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)
        return state, torch.round(val_out).item()

    def _placeholder_stop(self, args=None):
        return True

    def reset_to_state(self, state):
        self.features = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    # POSTCONDITIONS

    def _intervene_postcondition(self, init_state, current_state):

        obs = self.preprocessor.transform(pd.DataFrame.from_records([current_state]))
        return self.model.predict(obs)[0] >= 0.5

    def get_observation(self):

        obs = self.preprocessor_fare.transform(
            pd.DataFrame.from_records([self.features])
        )
        return torch.FloatTensor(obs.values[0])

    def get_cost(self, program_index, args_index):

        # Placeholder. Here we assume all actions have the same costs.
        return 2
