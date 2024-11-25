import numpy as np
import cv2 as cv

from src.base_step import BaseStep


class Binarizer(BaseStep):
    def __init__(self):
        super().__init__()

    def execute_step(self, img: np.ndarray):
        # Cut-off the bright details to focus on dark details
        _, thr_img = cv.threshold(img, 230, 0, cv.THRESH_TRUNC)
        # Normalize to create contrast
        cv.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        # Apply binary thresholding to create a black-and-white image
        _, binary_img = cv.threshold(thr_img, 240, 255, cv.THRESH_BINARY)

        # Ensure the output is normalized (optional for binary images, as values are 0 or 255)
        output_img = binary_img.copy()
        cv.normalize(binary_img, output_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        return output_img

