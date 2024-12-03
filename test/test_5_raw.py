import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
import os

images_path = 'E:\\hus-scanner\\test_images\\raw_png_imgs'

image = cv.imread(os.path.join(images_path, '0_raw.png'))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

corners = CornerPipeline(version="v1").execute(gray)

pipeline = Pipeline(stages=[
    RemoveShadow(),
    Binarizer()
])

gray = pipeline.execute(gray)


for point in corners:
    cv.circle(image, point, 10, (0, 255, 0), 10)

plt.imshow(image)
# plt.imshow(contoured_image, cmap="gray")
plt.show()
