import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
import os
from tqdm import tqdm

images_path = 'E:\\hus-scanner\\test_images\\test_jpgs'
corner_path = 'E:\\hus-scanner\\test_images\\jpg_scans'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(images_path, filename))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Binarizer()
        ])

        binary = pipeline.execute(gray)

        corners = CornerPipeline(version="v2").execute(gray)

        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_corner.png"), binary)
