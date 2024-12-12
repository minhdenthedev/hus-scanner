import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from src.base_step import BaseStep
# import os


class Enhancer(BaseStep):
    def __init__(self):
        super().__init__()
        # self.output_folder = "output"

    def execute_step(self, img: np.ndarray):
        blur = cv.GaussianBlur(img, (9, 9), 3)
        sharpened = cv.addWeighted(img, 3.0, blur, -2.0, 0)
        sharpened[sharpened > 200] = 255
        return sharpened
