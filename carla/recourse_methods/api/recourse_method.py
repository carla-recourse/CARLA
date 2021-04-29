from abc import ABC, abstractmethod


class RecourseMethod(ABC):
    @abstractmethod
    def get_counterfactuals(self, data, mlmodel):
        pass
