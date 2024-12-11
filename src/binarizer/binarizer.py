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
        _, thr_img = cv.threshold(img, 230, 0, cv.THRESH_TRUNC)
        # Normalize to create contrast
        cv.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        # _, binary_img = cv.threshold(thr_img, 240, 255, cv.THRESH_BINARY)

        # output_img = binary_img.copy()
        # cv.normalize(binary_img, output_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        # # Lưu ảnh vào thư mục output
        # if not os.path.exists(self.output_folder):
        #     os.makedirs(self.output_folder)
        # output_path = os.path.join(self.output_folder, "remove_shadow_output_img.jpg")
        # cv.imwrite(output_path, output_img)
        # print(f"Image saved to {output_path}")  # Thông báo vị trí lưu ảnh

        return thr_img

