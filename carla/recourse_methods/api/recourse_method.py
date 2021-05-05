from abc import ABC, abstractmethod

import pandas as pd


class RecourseMethod(ABC):
    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        pass
