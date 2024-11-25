from abc import ABC, abstractmethod
import numpy as np


class BaseStep(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute_step(self, img: np.ndarray):
        pass
