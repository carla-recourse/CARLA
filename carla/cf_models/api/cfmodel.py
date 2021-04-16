from abc import ABC, abstractmethod


class CFModel(ABC):
    @abstractmethod
    def get_counterfactuals(self, data, mlmodel):
        pass
