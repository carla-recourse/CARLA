from abc import ABC, abstractmethod


class Recourse_Method(ABC):
    @abstractmethod
    def get_counterfactuals(self, data, mlmodel):
        pass
