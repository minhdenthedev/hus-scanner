import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.utils import detect_corner

image = cv.imread("../test_images/IMG_8511.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = np.ones((7,7),np.uint8)
img = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel, iterations= 3)

# plt.imshow(img, cmap="gray")

pipeline = Pipeline(stages=[
    RemoveShadow(),
    Binarizer()
])

img = pipeline.execute(img)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
min_contour_area = 500  # Ngưỡng diện tích, có thể điều chỉnh tùy theo ảnh của bạn
large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
large_contours = sorted(large_contours, key=cv.contourArea, reverse=True)
for i in range(len(large_contours[0])):
    if 0 in large_contours[0][i][0]:
        large_contours.pop(0)

contoured_image = image.copy()
cv.drawContours(contoured_image, large_contours, -1, (0, 255, 0), 30)

plt.imshow(contoured_image)
plt.show()

