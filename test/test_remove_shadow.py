import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.utils import detect_contour
import os
from tqdm import tqdm

images_path = 'E:\\hus-scanner\\test_images\\unfiltered_pngs'
results_path = 'E:\\hus-scanner\\test_images\\contour_results'
binary_results_path = 'E:\\hus-scanner\\test_images\\binaries'


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

    cv.imwrite(os.path.join(binary_results_path, filename.split("_")[0] + '_bin.png'), img)

    contoured_image = detect_contour(img)

    return contoured_image


files = os.listdir(images_path)

files = [f for f in files if os.path.isfile(os.path.join(images_path, f))]

for filename in tqdm(files):
    img = contour_process(filename)
    number = filename.split("_")[0]
    new_file_name = number + "_contour.png"
    cv.imwrite(os.path.join(results_path, new_file_name), img)
