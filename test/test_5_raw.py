import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
import os

images_path = 'E:\\hus-scanner\\test_images\\raw_png_imgs'
corner_path = 'E:\\hus-scanner\\test_images\\corner_detection_v1'

list_images = os.listdir(images_path)

for filename in list_images:
    image = cv.imread(os.path.join(images_path, filename))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    corners = CornerPipeline(version="v1").execute(gray)

    for point in corners:
        cv.circle(image, point, 10, (0, 255, 0), 10)

    cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_corner.png"), image)
