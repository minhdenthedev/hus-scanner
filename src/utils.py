import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import pillow_heif


def detect_corner(img: np.ndarray):
    # # Bước 2: Chuyển sang grayscale
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Bước 3: Áp dụng Canny edge detection
    edges = cv.Canny(img, threshold1=50, threshold2=150)

    # Bước 4: Giãn các cạnh
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv.dilate(edges, kernel, iterations=1)

    # Bước 5: Tìm các contour và lọc theo kích thước
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]  # Giữ lại các contour lớn

    # Bước 6: Tìm contour lớn nhất (giả định là tờ giấy A4)
    largest_contour = max(contours, key=cv.contourArea)

    # Bước 7: Xấp xỉ contour để lấy đa giác có 4 cạnh
    epsilon = 0.02 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        return approx
    else:
        print(approx)


def batch_convert_to_png(input_folder, output_folder):
    # Register HEIC
    pillow_heif.register_heif_opener()

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        # If image is in HEIC format
        if filename.lower().endswith(".heic"):
            print(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")

            try:
                image = Image.open(input_path)

                image.save(output_path, format="PNG")
                print(f"Converted {filename} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")