import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import pillow_heif
import math
import cv2
import pytesseract


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
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c


def remove_nearly_parallel_lines(lines, min_distance, angle_threshold=np.deg2rad(5)):
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
            print(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{i}_raw.png")

            try:
                image = Image.open(input_path)

                image.save(output_path, format="PNG")
                print(f"Converted {filename} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def calculate_distance(point1, point2):
    """Tính khoảng cách giữa hai điểm."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
    return [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]

def find_top_2_largest_distances(points,width,height):
    """Tìm 2 đường chéo của tứ giác là 4 đỉnh của văn bản"""
    points = remove_out_of_bounds_points(points,width,height)
    if len(points) < 2:
        return []

    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i], points[j])
            distances.append(((points[i], points[j]), distance))

    distances.sort(key=lambda x: x[1], reverse=True)
    return distances[:2]

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
            quadrants['top_left']=(point)
        elif x >= width and y < height:
            quadrants['top_right']=(point)
        elif x < width and y >= height:
            quadrants['bottom_left']=(point)
        else:
            quadrants['bottom_right']=(point)

    corners = {
        'top_left': (0, 0),
        'top_right': (width*2, 0),
        'bottom_left': (0, height*2),
        'bottom_right': (width*2, height*2)
    }
    
    result = {}
    for quadrant, pts in quadrants.items():
        if pts:
            result[quadrant] = pts
        else:
            result[quadrant] = corners[quadrant]
    
    return result

# points = [(0, 0), (5, 5), (0, 5), (5, 0), (2, 3)]
# top_2_distances = find_top_2_largest_distances(points)
# for (point1, point2), distance in top_2_distances:
#     print(f"Cặp điểm: {point1}, {point2} - Khoảng cách: {distance:.2f}")

def detect_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Nhị phân hóa
    projection = np.sum(binary, axis=1)  # Tổng pixel theo chiều ngang
    return projection


def find_text_regions(projection, threshold=2000):
    regions = []
    start = None
    for i, value in enumerate(projection):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            regions.append((start, i))
            start = None
    return regions

def detect_lines_and_draw(binary,img, threshold=50):
    # Tính projection profile (tổng pixel theo chiều ngang)
    projection = np.sum(binary, axis=1)
    count=0
    
    # Tìm vùng chứa văn bản dựa trên ngưỡng
    start = None
    for i, value in enumerate(projection):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            # Vẽ một đường ngang tại vị trí dòng phát hiện được
            cv2.line(img, (0, start), (img.shape[1], start), (0, 255, 0), 1)
            cv2.line(img, (0, i), (img.shape[1], i), (0, 255, 0), 1)
            start = None
            count+=1
    print("số đường phát hiện được: ",count)
    return img, projection


def get_text_lines(img):
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    boxes = []
    for i in range(len(d['level'])):
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        boxes.append((x, y, x + w, y + h))
    return boxes

def correct_text_lines(img, boxes):
    h, w = img.shape[:2]
    src_points = []
    dst_points = []

    for (x1, y1, x2, y2) in boxes:
        src_points.append([x1, y1])
        src_points.append([x2, y2])
        dst_points.append([x1, y1])  # Giữ nguyên y
        dst_points.append([x2, y1])  # Làm phẳng theo y

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    M = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
    corrected_img = cv2.warpAffine(img, M, (w, h))
    return corrected_img

