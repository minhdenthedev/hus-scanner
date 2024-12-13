from typing import List
import numpy as np

from src.base_step import BaseStep



class Pipeline:
    def __init__(self, stages: List[BaseStep]):
        self.stages = stages

    def execute(self, img: np.ndarray):
        # Truyền dữ liệu giữa các bước
        for stage in self.stages:
            if isinstance(img, tuple):  # Kiểm tra nếu img là tuple (bao gồm img và vertices)
                img, vertices = stage.execute_step(*img)  # Truyền cả img và vertices
            else:
                img = stage.execute_step(img)  # Chỉ truyền ảnh nếu không có dữ liệu bổ sung
        return img
