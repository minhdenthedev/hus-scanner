import numpy as np
import cv2 as cv

from src.base_step import BaseStep
# import os


class Binarizer(BaseStep):
    def __init__(self):
        super().__init__()
        # self.output_folder = "output"

    def execute_step(self, img: np.ndarray):
        # Cut-off the bright details to focus on dark details
        _, thr_img = cv.threshold(img, 200, 0, cv.THRESH_TRUNC)
        # Normalize to create contrast
        cv.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        _, binary_img = cv.threshold(thr_img, 240, 255, cv.THRESH_BINARY)

        output_img = binary_img.copy()
        cv.normalize(binary_img, output_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        return binary_img

