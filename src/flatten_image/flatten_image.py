import cv2
import numpy as np
from src.base_step import BaseStep
from PIL import Image

class FlattenImage(BaseStep):
    def __init__(self, approx: np.ndarray):
        """
        Constructor to initialize the vertices (approx) for flattening.
        :param approx: np.ndarray - Array of vertices to warp the image.
        """
        super().__init__()
        self.approx = approx

    def execute_step(self, img: np.ndarray):
        """
        Main method to flatten the image.
        :param img: np.ndarray - Input image to be flattened.
        :return: Flattened image.
        """
        if img is None or self.approx is None:
            print("Error: Image or vertices (approx) is None.")
            return img
        
        # Flatten the image using perspective transform
        flattened_img = self.flatten(img)
        return flattened_img

    def reorder(self):
        """
        Reorder the vertices to ensure consistent order:
        [top-left, top-right, bottom-right, bottom-left].
        :return: Reordered vertices.
        """
        vertices = []
        for point in self.approx:
            vertices.append(list(point[0]))
        vertices = np.array(vertices, dtype=np.float32)

        reordered = np.zeros_like(vertices, dtype=np.float32)
        add = vertices.sum(axis=1)  # Sum x + y
        diff = np.diff(vertices, axis=1)  # Difference x - y

        # Sort vertices
        reordered[0] = vertices[np.argmin(add)]  # Top-left
        reordered[2] = vertices[np.argmax(add)]  # Bottom-right
        reordered[1] = vertices[np.argmin(diff)]  # Top-right
        reordered[3] = vertices[np.argmax(diff)]  # Bottom-left

        return reordered

    def flatten(self, img: np.ndarray):
        """
        Apply perspective transformation to flatten the image.
        :param img: np.ndarray - Input image.
        :return: Warped (flattened) image.
        """
        # Reorder vertices
        vertices = self.reorder()
        (a, b, c, d) = vertices

        # Compute width and height of the new flattened image
        width1 = np.linalg.norm(c - d)
        width2 = np.linalg.norm(b - a)
        height1 = np.linalg.norm(b - c)
        height2 = np.linalg.norm(a - d)

        max_width = max(int(width1), int(width2))
        max_height = max(int(height1), int(height2))

        # Destination points for the warped rectangle
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(vertices, dst)
        warped_img = cv2.warpPerspective(img, M, (max_width, max_height))

        return warped_img
