from typing import List, Tuple

import cv2 as cv
import numpy as np

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.utils import polar_to_cartesian, find_intersection, detect_contour, remove_nearly_parallel_lines, \
    remove_parallel_v2, show, calculate_distance, show_two, line_through_two_points, check_angle_in_range
import matplotlib.pyplot as plt
import networkx as nx


def remove_out_of_bounds_points(points, width, height):
    """
    Loại bỏ các điểm ngoài giới hạn tọa độ.

    Args:
        points (list of tuple): Danh sách các điểm (x, y).
        width (int): Ngưỡng tối đa cho tọa độ x (chiều rộng ảnh).
        height (int): Ngưỡng tối đa cho tọa độ y (chiều cao ảnh).

    Returns:
        list of tuple: Danh sách các điểm hợp lệ.
    """

    result = [(x, y) for x, y in points
              if (-width * 0.1) <= x < (width * 1.1) and (-0.1 * height) <= y < (1.1 * height)]
    return result


def find_top_2_largest_distances(points, width, height):
    """Tìm 2 đường chéo của tứ giác là 4 đỉnh của văn bản"""
    points = remove_out_of_bounds_points(points, width, height)
    if len(points) < 2:
        return []

    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i], points[j])
            distances.append(((points[i], points[j]), distance))

    distances.sort(key=lambda x: x[1], reverse=True)
    top_2_distances = distances[:2]
    vertices = []
    for (point1, point2), distance in top_2_distances:
        vertices.append(point1)
        vertices.append(point2)
    return vertices


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
    polys.pop(0)
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


def sort_points_clockwise(points):
    points = np.array(points)

    # Tìm trọng tâm (cx, cy)
    centroid = np.mean(points, axis=0)
    cx, cy = centroid

    # Tính góc của từng điểm so với trọng tâm
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)

    # Sắp xếp các điểm dựa trên góc theo thứ tự giảm dần
    sorted_indices = np.argsort(-angles)
    sorted_points = points[sorted_indices]

    return sorted_points.tolist()


def corner_detection_v2(img: np.ndarray):
    gray_for_contour = img.copy()

    # Xoa het chu
    kernel = np.ones((7, 7), np.uint8)
    gray_for_contour = cv.morphologyEx(gray_for_contour, cv.MORPH_CLOSE, kernel, iterations=3)

    # Chuyen ve nhi phan
    pipeline = Pipeline(stages=[
        RemoveShadow(),
        Binarizer()
    ])
    gray_for_contour = pipeline.execute(gray_for_contour)
    edges = cv.Canny(gray_for_contour, 50, 200)
    edges = cv.dilate(edges, np.ones((5, 5)))
    polys = detect_contour(edges)
    contoured_image = gray_for_contour.copy()
    contoured_image[:] = 0
    for poly in polys:
        cv.drawContours(contoured_image, [poly], -1, (255, 255, 255), 1)
    edges = cv.Canny(contoured_image, 150, 250)
    edges = cv.dilate(edges, kernel=np.ones((3, 3)))
    lines = cv.HoughLines(edges, 1, np.pi / 180, 260)

    lines = remove_parallel_v2([line[0] for line in lines], min(img.shape[0], img.shape[1]) * 0.4)
    copy = edges.copy()
    copy[:] = 0
    for line in lines:
        rho, theta = line  # Extract rho and theta

        # Convert rho and theta to Cartesian coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Calculate the start and end points of the line
        x1 = int(x0 + 1000 * (-b))  # Extend line in one direction
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))  # Extend line in the other direction
        y2 = int(y0 - 1000 * (a))

        # Draw the line
        cv.line(copy, (x1, y1), (x2, y2), (255, 255, 255), 2)
    lines = cv.HoughLines(copy, 1, np.pi / 180, 260)

    line_equations = []

    G = nx.Graph()

    for i, line in enumerate(lines):
        rho, theta = line[0]
        line_equations.append(polar_to_cartesian(rho, theta))
        G.add_node(i)

    for i in range(len(line_equations)):
        for j in range(i + 1, len(line_equations)):
            if check_angle_in_range(line_equations[i], line_equations[j], min_angle=60, max_angle=130):
                G.add_edge(i, j)

    isolated_nodes = [node for node in G.nodes if G.degree(node) < 2]
    for index in isolated_nodes:
        line_equations.pop(index)


    intersections = []
    for i in range(len(line_equations)):
        for j in range(i + 1, len(line_equations)):
            intersection = find_intersection(line_equations[i], line_equations[j])
            if intersection:
                intersections.append(intersection)

    intersections = remove_out_of_bounds_points(intersections, img.shape[1], img.shape[0])

    if len(intersections) > 4:
        intersections = refine_corners(img, intersections)

    return intersections


def refine_corners(img: np.ndarray, corners: List[Tuple]):
    image = img.copy()
    image[:] = 0
    corners = sort_points_clockwise(corners)
    lines = []
    last_corner = corners[0]
    for corner in corners[1:]:
        lines.append(line_through_two_points(last_corner, corner))
        last_corner = corner
    lines.append(line_through_two_points(last_corner, corners[0]))
    line_equations = [lines[0]]
    last_line = lines[0]
    for line in lines[1:]:
        if check_angle_in_range(last_line, line, min_angle=70, max_angle=130):
            line_equations.append(line)
            last_line = line
        else:
            last_line = line
            continue
    if not check_angle_in_range(last_line, lines[0], min_angle=70, max_angle=130):
        line_equations.pop(0)
    intersections = []
    for i in range(len(line_equations)):
        for j in range(i + 1, len(line_equations)):
            intersection = find_intersection(line_equations[i], line_equations[j])
            if intersection:
                intersections.append(intersection)

    intersections = remove_out_of_bounds_points(intersections, img.shape[1], img.shape[0])

    return intersections


class CornerPipeline(Pipeline):
    def __init__(self, stages=None, version: str = 'v1'):
        super().__init__(stages=stages)
        self.version = version

    def execute(self, img: np.ndarray):
        if self.version == "v1":
            return corner_detection_v1(img)
        elif self.version == "v2":
            return corner_detection_v2(img)
