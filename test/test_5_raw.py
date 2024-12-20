import cv2 as cv
import numpy as np

from src.binarizer.remove_shadow import RemoveShadow
from src.corner_detector.corner_pipeline import CornerPipeline
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
        if filename != "29_raw.png":
            continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find corners
        corners = CornerPipeline(version="v2").execute(gray)
        for corner in corners:
            cv.circle(image, corner, 20, (0, 255, 0), 20)
        show(image)

        if len(corners) == 4:
            # Warping
            approx = np.array(corners, dtype=np.float32).reshape((-1, 1, 2))
            warped_image = Warping(approx).execute_step(image)
            warped_image = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
            result = warped_image
        else:
            result = gray

        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Enhancer()
        ])
        result = pipeline.execute(result)

        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_scanned.png"), result)
