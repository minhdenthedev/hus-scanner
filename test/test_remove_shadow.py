import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.utils import detect_corner

img = cv.imread("../test_images/IMG_8511.jpg", 0)

pipeline = Pipeline(stages=[
    RemoveShadow(),
    Binarizer()
])

output = pipeline.execute(img)

corners = detect_corner(img)
print(corners)

