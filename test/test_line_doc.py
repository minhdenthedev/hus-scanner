import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
from src.warping.warping import Warping
from src.find_contours.find_contours import FindContours
from src.utils import detect_lines, get_text_lines, correct_text_lines
import os
from tqdm import tqdm

images_path = '.\\test_images\\unfiltered_pngs'
warped_path = '.\\test_images\\line_detection'
binary_path = '.\\test_images\\binary_detection'
boundary_path = '.\\test_images\\boundary'

list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        # Load the image
        image = cv.imread(os.path.join(images_path, filename))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Binarizer(),
            FindContours()
        ])

        contours = pipeline.execute(gray)

        # Create a copy of the binary image to draw contours on
        boundary_image = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored drawing

        # Draw the contours on the boundary image
        cv.drawContours(boundary_image, contours, -1, (0, 255, 0), 2)  # Green color, thickness 2

        # Save the image with drawn contours
        cv.imwrite(os.path.join(boundary_path, filename.split("_")[0] + "bounded.png"), boundary_image)