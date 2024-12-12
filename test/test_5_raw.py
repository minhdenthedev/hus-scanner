import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.remove_shadow import RemoveShadow

from src.enhancer.enhancer import Enhancer
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
from src.warping.warping import Warping
from src.utils import find_top_2_largest_distances
import os
from tqdm import tqdm

images_path = 'E:\\hus-scanner\\test_images\\raw_png_imgs'
corner_path = 'E:\\hus-scanner\\test_images\\contour_results'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(images_path, filename))
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find corners
        corners = CornerPipeline(version="v2").execute(gray)
        height, width = gray.shape
        vertices = find_top_2_largest_distances(corners, width, height)
        if len(vertices) != 4:
            print(f"Error at file {filename}")
            continue

        # Warping
        approx = np.array(vertices, dtype=np.float32).reshape((-1, 1, 2))
        warped_image = Warping(approx).execute_step(image)
        warped_image = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)

        # Result
        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Enhancer()
        ])

        result = pipeline.execute(warped_image)

        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_warped.png"), result)
