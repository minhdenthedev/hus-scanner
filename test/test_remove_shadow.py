import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.utils import detect_corner
import os
from tqdm import tqdm


images_path = 'E:\\hus-scanner\\test_images\\Processed_JPG'
results_path = 'E:\\hus-scanner\\test_images\\contour_results'


def contour_process(filename):
    filepath = os.path.join(images_path, filename)
    image = cv.imread(filepath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((7, 7), np.uint8)
    img = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel, iterations=3)

    # plt.imshow(img, cmap="gray")

    pipeline = Pipeline(stages=[
        RemoveShadow(),
        Binarizer()
    ])

    img = pipeline.execute(img)

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000  # Ngưỡng diện tích, có thể điều chỉnh tùy theo ảnh của bạn
    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    large_contours = sorted(large_contours, key=cv.contourArea, reverse=True)
    large_contours.pop(0)
    contoured_image = image.copy()

    for contour in large_contours:
        # Tính chu vi của contour
        epsilon = 0.02 * cv.arcLength(contour, True)  # epsilon là 2% chu vi của contour
        approx = cv.approxPolyDP(contour, epsilon, True)  # Xấp xỉ contour thành đa giác

        # Vẽ contour gốc
        cv.drawContours(contoured_image, [contour], -1, (0, 255, 0), 30)
        cv.drawContours(contoured_image, [approx], -1, (0, 0, 255), 30)

    # contoured_image = image.copy()
    # cv.drawContours(contoured_image, large_contours, -1, (0, 255, 0), 30)

    return contoured_image


files = os.listdir(images_path)

files = [f for f in files if os.path.isfile(os.path.join(images_path, f))]

for filename in tqdm(files):
    img = contour_process(filename)
    number = filename.split("_")[0]
    new_file_name = number + "_contour.jpg"
    cv.imwrite(os.path.join(results_path, new_file_name), img)
