import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import pillow_heif


def detect_contour(image: np.ndarray):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    large_contours = sorted(contours, key=cv.contourArea, reverse=True)
    large_contours.pop(0)

    polys = []

    for contour in large_contours[0:4]:
        epsilon = 0.02 * cv.arcLength(contour, True)  # epsilon là 2% chu vi của contour
        approx = cv.approxPolyDP(contour, epsilon, True)  # Xấp xỉ contour thành đa giác
        polys.append(approx)

    return polys


# Function to convert (rho, theta) to line equation coefficients (a, b, c)
def polar_to_cartesian(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c


# Function to find intersection of two lines (a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0)
def find_intersection(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:  # Lines are parallel
        return None
    x = (b1 * c2 - b2 * c1) / determinant
    y = (a2 * c1 - a1 * c2) / determinant
    return (int(x), int(y))


def batch_convert_to_png(input_folder, output_folder):
    # Register HEIC
    pillow_heif.register_heif_opener()

    os.makedirs(output_folder, exist_ok=True)

    for i, filename in enumerate(os.listdir(input_folder)):
        # If image is in HEIC format
        if filename.lower().endswith(".heic"):
            print(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{i}_raw.png")

            try:
                image = Image.open(input_path)

                image.save(output_path, format="PNG")
                print(f"Converted {filename} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
