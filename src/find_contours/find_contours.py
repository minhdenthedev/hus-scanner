import numpy as np
from src.base_step import BaseStep
import cv2 as cv

class FindContours(BaseStep):
    def execute_step(self, binary_image: np.ndarray):
        """
        Find contours for the words in the binary image.
        Only returns contours corresponding to black words.
        """
        # Invert the binary image because words are black
        inverted_binary = cv.bitwise_not(binary_image)
        
        # Find contours in the inverted binary image
        contours, _ = cv.findContours(inverted_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Return the contours
        return contours
