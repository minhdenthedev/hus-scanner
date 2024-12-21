import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
from src.evaluate.evaluate import calculate_overall_score
import os
from tqdm import tqdm

images_path = 'E:\\hus-scanner\\test_images\\unfiltered_pngs'
corner_path = 'E:\\hus-scanner\\test_images\\corner_detection_v2'
warped_path = '.\\test_images\\warped'

list_images = os.listdir(images_path)
list_warped_images = os.listdir(warped_path)

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

        for point in corners:
            cv.circle(image, point, 20, (0, 255, 0), 20)

        cv.imwrite(os.path.join(corner_path, filename.split("_")[0] + "_corner.png"), binary)

    # evaluate warped images
    for filename in tqdm(list_warped_images):
        image = cv.imread(os.path.join(warped_path, filename))
        
        print("Test for", filename, ":")
        score = calculate_overall_score(image)
        print(f"Overall score for {filename}: {score}\n")

        print("Test score for", filename, ":")
        score = calculate_overall_score(image)
        print(f"Overall score for {filename}: {score}\n")