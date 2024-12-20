import cv2 as cv
import numpy as np

from src.binarizer.remove_shadow import RemoveShadow
from src.corner_detector.corner_pipeline import CornerPipeline, sort_points_clockwise, refine_corners
from src.enhancer.enhancer import Enhancer
from src.pipeline import Pipeline
import os
from tqdm import tqdm
from src.utils import show_two, show

from src.warping.warping import Warping

images_path = 'E:\\hus-scanner\\test_images\\evaluate_images'
corner_path = 'E:\\hus-scanner\\test_images\\evaluate_output'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(images_path, filename))
        print(f"File name: {filename}")
        # if filename != "7_raw.png":
        #     continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find corners
        corners = CornerPipeline(version="v2").execute(gray)
        # for corner in corners:
        #     cv.circle(image, corner, 10, (0, 255, 0), 20)
        # show(image)
        result = None

        if len(corners) == 4:
            # Warping
            approx = np.array(corners, dtype=np.float32).reshape((-1, 1, 2))
            warped_image = Warping(approx).execute_step(image)
            warped_image = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
            result = warped_image
        else:
            print(f"Error: {filename}")
            result = gray

        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Enhancer()
        ])
        result = pipeline.execute(result)

        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_scanned.png"), result)
