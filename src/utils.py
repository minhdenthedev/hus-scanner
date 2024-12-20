import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import pillow_heif
import math


def show_two(img1: np.ndarray, img2: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap='gray' if len(img1.shape) == 2 else None)
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray' if len(img2.shape) == 2 else None)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def detect_contour(image: np.ndarray):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000
    large_contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]
    large_contours = sorted(large_contours, key=cv.contourArea, reverse=True)

    polys = []

    for contour in large_contours:
        epsilon = 0.02 * cv.arcLength(contour, True)  # epsilon là 2% chu vi của contour
        approx = cv.approxPolyDP(contour, epsilon, True)  # Xấp xỉ contour thành đa giác
        polys.append(approx)

    return polys


# Function to convert (rho, theta) to line equation coefficients (a, b, c)
def polar_to_cartesian(rho, theta):
    a = rho * np.cos(theta)
    b = rho * np.sin(theta)
    c = - (rho * rho)
    return a, b, c


def remove_nearly_parallel_lines(lines, min_distance, angle_threshold=np.deg2rad(15)):
    filtered_lines = []

    for i, (rho1, theta1) in enumerate(lines):
        is_parallel = False
        for rho2, theta2 in filtered_lines:
            # Calculate the absolute angle difference, normalized to [0, π]
            angle_diff = abs(theta1 - theta2)
            angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle wraparound
            distance = abs(rho2 - rho1)
            if angle_diff < angle_threshold and distance < min_distance:
                is_parallel = True
                break

        if not is_parallel:
            filtered_lines.append((float(rho1), float(theta1)))

    return filtered_lines


def remove_parallel_v2(lines, min_distance, angle_threshold=np.deg2rad(15)):
    filtered_lines = []
    for i, (rho1, theta1) in enumerate(lines):
        is_parallel = False
        x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)

        for rho2, theta2 in filtered_lines:
            x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)

            # Tính chênh lệch góc và khoảng cách
            angle_diff = abs(theta1 - theta2)
            angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle wraparound
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if angle_diff < angle_threshold and distance < min_distance:
                is_parallel = True
                break

        if not is_parallel:
            filtered_lines.append((float(rho1), float(theta1)))
        # if len(filtered_lines) >= 4:
        #     return filtered_lines

    return filtered_lines


def is_line_within_image(a, b, c, width, height):
    intersections = []

    # Cạnh dưới (y = 0)
    if a != 0:
        x = -c / a
        if 0 <= x <= width:
            intersections.append((x, 0))

    # Cạnh trên (y = height)
    if a != 0:
        x = -(b * height + c) / a
        if 0 <= x <= width:
            intersections.append((x, height))

    # Cạnh trái (x = 0)
    if b != 0:
        y = -c / b
        if 0 <= y <= height:
            intersections.append((0, y))

    # Cạnh phải (x = width)
    if b != 0:
        y = -(a * width + c) / b
        if 0 <= y <= height:
            intersections.append((width, y))

    # Nếu có giao điểm nào nằm trong hình ảnh, đường thẳng nằm trong ảnh
    return len(intersections) > 0


# Function to find intersection of two lines (a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0)
def find_intersection(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:  # Lines are parallel
        return None
    x = (b1 * c2 - b2 * c1) / determinant
    y = (a2 * c1 - a1 * c2) / determinant
    return int(x), int(y)


def batch_convert_to_png(input_folder, output_folder):
    # Register HEIC
    pillow_heif.register_heif_opener()

    os.makedirs(output_folder, exist_ok=True)

    for i, filename in enumerate(os.listdir(input_folder)):
        # If image is in HEIC format
        if filename.lower().endswith(".heic"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{i}_raw.png")

            try:
                image = Image.open(input_path)

                image.save(output_path, format="PNG")
                print(f"Converted {filename} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
        elif filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{i}_raw.png")
            try:
                image = cv.imread(input_path)
                cv.imwrite(output_path, image)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


def calculate_distance(point1, point2):
    """Tính khoảng cách giữa hai điểm."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def show(img: np.ndarray):
    plt.imshow(img, cmap="gray")
    plt.show()


def fill_image_verticles(center_point, points):
    height, width = center_point
    quadrants = {
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }

    for point in points:
        x, y = point
        if x < width and y < height:
            quadrants['top_left'] = (point)
        elif x >= width and y < height:
            quadrants['top_right'] = (point)
        elif x < width and y >= height:
            quadrants['bottom_left'] = (point)
        else:
            quadrants['bottom_right'] = (point)

    corners = {
        'top_left': (0, 0),
        'top_right': (width * 2, 0),
        'bottom_left': (0, height * 2),
        'bottom_right': (width * 2, height * 2)
    }

    result = {}
    for quadrant, pts in quadrants.items():
        if pts:
            result[quadrant] = pts
        else:
            result[quadrant] = corners[quadrant]

    return result
