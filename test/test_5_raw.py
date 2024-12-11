import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
from src.warping.warping import Warping
from src.utils import find_top_2_largest_distances
import os
from tqdm import tqdm

images_path = 'E:\\hus-scanner\\test_images\\test_jpgs'
corner_path = 'E:\\hus-scanner\\test_images\\jpg_scans'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(images_path, filename))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        height, width = gray.shape


        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Binarizer()
        ])

        binary = pipeline.execute(gray)

        corners = CornerPipeline(version="v2").execute(gray)
        # points = [(0, 0), (5, 5), (0, 5), (5, 0), (2, 3)]
        top_2_distances = find_top_2_largest_distances(corners, width, height)
        print()
        verticles = []
        print("file: ",filename.split("_")[0])
        for (point1, point2), distance in top_2_distances:
            verticles.append(point1)
            verticles.append(point2)
            print(f"Cặp điểm: {point1}, {point2} - Khoảng cách: {distance:.2f}")


        for point in verticles:
            cv.circle(image, point, 20, (0, 255, 0), 20)

        approx = np.array(verticles, dtype=np.float32).reshape((-1, 1, 2))
        warping_only = Pipeline(stages=[
            Warping(approx)
        ])
        # print("approx",approx)

        if len(verticles) != 4:
            print(f"Invalid number of vertices ({len(verticles)}) for warping in {filename}.")
            continue
        warped_image = warping_only.execute(image)
        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_warped.png"), warped_image)
