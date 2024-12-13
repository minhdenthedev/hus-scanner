import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.remove_shadow import RemoveShadow

from src.enhancer.enhancer import Enhancer
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline, find_top_2_largest_distances
from src.warping.warping import Warping
from src.exceptions.not_good_enough import NotGoodEnoughException
from src.utils import show
import os
from tqdm import tqdm

images_path = 'E:\\hus-scanner\\test_images\\raw_png_imgs'
corner_path = 'E:\\hus-scanner\\test_images\\contour_results'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        # if filename != "20f2cdfbc4547e0a27453.jpg":
        #     continue
        image = cv.imread(os.path.join(images_path, filename))
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find corners
        corners = CornerPipeline(version="v2").execute(gray)
        if len(corners) != 4:
            raise NotGoodEnoughException(details=f"{filename}")

        # Warping
        approx = np.array(corners, dtype=np.float32).reshape((-1, 1, 2))
        warped_image = Warping(approx).execute_step(image)
        warped_image = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)

        # Result
        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Enhancer()
        ])

        result = pipeline.execute(warped_image)
        # show(result)

        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_warped.png"), result)
