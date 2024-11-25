from typing import List
import numpy as np

from src.base_step import BaseStep


class Pipeline:
    def __init__(self, stages: List[BaseStep]):
        self.stages = stages

    def execute(self, img: np.ndarray):
        for stage in self.stages:
            img = stage.execute_step(img)
        return img
