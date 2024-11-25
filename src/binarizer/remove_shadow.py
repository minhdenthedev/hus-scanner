import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.base_step import BaseStep


class RemoveShadow(BaseStep):
    def __init__(self, dilate_kernel_size: int = 7, median_blur_kernel_size: int = 21):
        super().__init__()
        self.dilate_kernel_size = dilate_kernel_size
        self.median_blur_kernel_size = median_blur_kernel_size

    def execute_step(self, img: np.ndarray):
        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        dilated_image = cv.dilate(img, kernel)

        bg_img = cv.medianBlur(dilated_image, self.median_blur_kernel_size)

        diff_img = 255 - cv.absdiff(img, bg_img)

        norm_img = diff_img.copy()
        # Normalize image to [0, 255]
        cv.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        return norm_img

