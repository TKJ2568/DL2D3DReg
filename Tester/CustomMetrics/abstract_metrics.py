from abc import ABC, abstractmethod


class AbstractMetrics(ABC):

    @abstractmethod
    def __call__(self, tru, pre):
        pass