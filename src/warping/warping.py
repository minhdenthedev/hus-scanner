import cv2
import numpy as np

from src.base_step import BaseStep


class Warping(BaseStep):
    def __init__(self, approx: np.ndarray):
        super().__init__()
        self.approx = approx

    def execute_step(self, img: np.ndarray):
        if img is None or self.approx is None:
            # Handle the case where img or approx is None
            print("Error: img or approx is None.")
            return img, self.approx
        
        # Call crop_out to process the image and obtain the cropped image
        cropped_img = self.crop_out(img)
        # Return both cropped image and vertices
        return cropped_img
    
    def reorder(self):
        vertices = []
        for i in self.approx:
            vertices.append(list(i[0]))
        # Convert to NumPy array (if not already)
        vertices = np.array(vertices, dtype=np.float32)
        
        reordered = np.zeros_like(vertices, dtype=np.float32)
        add = vertices.sum(axis=1)  # Sum x + y
        diff = np.diff(vertices, axis=1)  # Difference x - y

        # Sort vertices: [top-left, top-right, bottom-right, bottom-left]
        reordered[0] = vertices[np.argmin(add)]  # Top-left
        reordered[2] = vertices[np.argmax(add)]  # Bottom-right
        reordered[1] = vertices[np.argmin(diff)]  # Top-right
        reordered[3] = vertices[np.argmax(diff)]  # Bottom-left

        return reordered

    def crop_out(self, im):
        # Get the reordered vertices
        vertices = self.reorder()
        (a, b, c, d) = vertices

        # Calculate width and height of the warped rectangle
        w1 = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2))
        w2 = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))
        h1 = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2))
        h2 = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2))

        max_width = max(int(w1), int(w2))
        max_height = max(int(h1), int(h2))

        # Destination coordinates for the warped rectangle
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Compute perspective transform and apply it
        M = cv2.getPerspectiveTransform(vertices, dst)
        cropped = cv2.warpPerspective(im, M, (max_width, max_height))
        
        return cropped
