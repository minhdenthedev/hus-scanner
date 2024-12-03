import cv2 as cv
import numpy as np

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.utils import polar_to_cartesian, find_intersection, detect_contour
import matplotlib.pyplot as plt


def corner_detection_v1(img: np.ndarray):
    gray_for_contour = img.copy()
    kernel = np.ones((7, 7), np.uint8)
    gray_for_contour = cv.morphologyEx(gray_for_contour, cv.MORPH_CLOSE, kernel, iterations=3)
    pipeline = Pipeline(stages=[
        RemoveShadow(),
        Binarizer()
    ])
    gray_for_contour = pipeline.execute(gray_for_contour)

    polys = detect_contour(gray_for_contour)

    contoured_image = gray_for_contour.copy()
    contoured_image[:] = 0

    for poly in polys:
        cv.drawContours(contoured_image, [poly], -1, (255, 255, 255), 5)
    contoured_image = cv.dilate(contoured_image, np.ones((25, 25)))
    contoured_image = cv.erode(contoured_image, np.ones((25, 25)))

    edges = cv.Canny(contoured_image, 50, 150, apertureSize=3)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 260)

    line_equations = []

    for line in lines:
        rho, theta = line[0]
        line_equations.append(polar_to_cartesian(rho, theta))

    intersections = []
    for i in range(len(line_equations)):
        for j in range(i + 1, len(line_equations)):
            intersection = find_intersection(line_equations[i], line_equations[j])
            if intersection:
                intersections.append(intersection)

    return intersections


def corner_detection_v2(img: np.ndarray):
    gray_for_contour = img.copy()
    kernel = np.ones((7, 7), np.uint8)
    gray_for_contour = cv.morphologyEx(gray_for_contour, cv.MORPH_CLOSE, kernel, iterations=3)
    pipeline = Pipeline(stages=[
        RemoveShadow(),
        Binarizer()
    ])
    gray_for_contour = pipeline.execute(gray_for_contour)
    plt.imshow(gray_for_contour, cmap="gray")
    # Detect corners using goodFeaturesToTrack
    max_corners = 4  # Maximum number of corners to detect
    quality_level = 0.7  # Minimum accepted quality of corners
    min_distance = 200  # Minimum distance between corners

    corners = cv.goodFeaturesToTrack(gray_for_contour, max_corners, quality_level, min_distance)
    corners = [(int(corner[0][0]), int(corner[0][1])) for corner in corners]
    return corners


class CornerPipeline(Pipeline):
    def __init__(self, stages=None, version: str = 'v1'):
        super().__init__(stages=stages)
        self.version = version

    def execute(self, img: np.ndarray):
        if self.version == "v1":
            return corner_detection_v1(img)
        elif self.version == "v2":
            return corner_detection_v2(img)


