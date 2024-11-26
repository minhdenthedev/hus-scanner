import numpy as np
import cv2 as cv

from src.base_step import BaseStep

from src.base_step import BaseStep
import cv2 as cv
import numpy as np

class CornerDetector(BaseStep):
    def __init__(self):
        super().__init__()

    def execute_step(self, img: np.ndarray):
        # Your existing logic to detect corners
        img, approx = self.detect_corner(img)
        return img, approx

    def detect_corner(self, img: np.ndarray):
        # Bước 3: Áp dụng Canny edge detection
        edges = cv.Canny(img, threshold1=50, threshold2=150)

        # Bước 4: Giãn các cạnh
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv.dilate(edges, kernel, iterations=1)

        # Bước 5: Tìm các contour và lọc theo kích thước
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]  # Giữ lại các contour lớn

        # Bước 6: Tìm contour lớn nhất (giả định là tờ giấy A4)
        largest_contour = max(contours, key=cv.contourArea)

        # Bước 7: Xấp xỉ contour để lấy đa giác có 4 cạnh
        epsilon = 0.02 * cv.arcLength(largest_contour, True)
        approx = cv.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:
            return img, approx
        else:
            print("wrong number of approx >4 corners")
            return img, None
