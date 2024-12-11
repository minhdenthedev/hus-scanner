import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
from src.warping.warping import Warping
from src.utils import find_top_2_largest_distances, fill_image_verticles
import os
from tqdm import tqdm

images_path = '.\\test_images\\unfiltered_pngs'
corner_path = '.\\test_images\\corner_detection_v2'
warped_path = '.\\test_images\\warped'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(images_path, filename))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        center_point = (height/2, width/2)

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


        # Fill verticle if need 
        if len(verticles) != 4:
            print(f"Invalid number of vertices ({len(verticles)}) for warping in {filename}.Starting auto fill:")
            verticles_dict=fill_image_verticles(center_point,verticles)
            verticles =  [value for key, value in verticles_dict.items()]
            print('vertices:', verticles)
        # Points to approx 
        approx = np.array(verticles, dtype=np.float32).reshape((-1, 1, 2))


        # Warping work
        warping_only = Pipeline(stages=[
            Warping(approx)
        ])
        warped_image = warping_only.execute(image)

        cv.imwrite(os.path.join(warped_path, filename.split("_")[0] + "_warped.png"), warped_image)
