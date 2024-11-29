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
        img, approx = self.detect_corner(img)
        if approx is None:
            print("Error: No valid corners found.")
        return img, approx

    def detect_corner(self, img: np.ndarray):
        # Step 3: Apply Canny edge detection
        edges = cv.Canny(img, threshold1=50, threshold2=150)

        # Step 4: Dilate the edges
        kernel = np.ones((9, 9), np.uint8)
        dilated = cv.dilate(edges, kernel, iterations=1)

        # Step 5: Find contours and filter by size
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]  # Keep large contours

        # Step 6: Find the largest contour (assuming it's the paper)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)

            # Step 7: Approximate the contour to get a polygon with 4 edges
            epsilon = 0.02 * cv.arcLength(largest_contour, True)
            approx = cv.approxPolyDP(largest_contour, epsilon, True)

            # Return corners if it's a 4-sided polygon
            if len(approx) == 4:
                return img, approx

        print("wrong number of approx >4 corners")
        return img, None  # Return img and None if no valid corners are found