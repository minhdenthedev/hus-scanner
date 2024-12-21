import cv2
import numpy as np
from pytesseract import image_to_data
from pytesseract.pytesseract import Output
import pytesseract
from pytesseract import Output
import math

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Adjust path if to tesseract.exe

def detect_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mean, stddev = cv2.meanStdDev(laplacian)

    variance = stddev[0][0] ** 2
    #print("Variance:", variance)

    threshold = 200
    return variance < threshold

def detect_over_exposure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prepare variables
    hist_size = [256]
    ranges = [0, 255]
    channels = [0]
    accumulate = False
    
    # Calculate histogram
    hist = cv2.calcHist([gray], channels, None, hist_size, ranges)
    
    # Get min/max values from the histogram
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    
    # If the maximum location y-value is greater than 240
    if max_loc[1] > 240:
        # Get the histogram data
        data = hist.flatten()
        
        # Calculate the number of dark pixels
        dark_pixels = np.sum(data[:201])
        
        # Get total pixels
        total_pixels = img.shape[0] * img.shape[1]
        
        # Calculate the percentage of dark pixels
        percent = dark_pixels / total_pixels
        
        print(percent)
        
        # If the percentage of dark pixels is less than 20%, return True (indicating a good image)
        if percent < 0.2:
            return True
        else:
            return False
    else:
        return False

def get_ocr_confidence(img):
    # Sử dụng pytesseract để lấy dữ liệu OCR
    ocr_result = pytesseract.image_to_data(img, output_type=Output.DICT)
    
    # Tính toán độ chính xác trung bình từ các dòng OCR
    confidences = ocr_result['conf']
    valid_confidences = [conf for conf in confidences if conf != -1]
    
    if valid_confidences:
        avg_confidence = sum(valid_confidences) / len(valid_confidences)
        # print("OCR Confidence:", avg_confidence)
        return avg_confidence
    return 0


def detect_aspect_ratio_incorrect(img, document_type=None):
    h, w = img.shape[:2]

    if document_type is None or document_type == "None":
        return False

    doc_width, doc_height = map(float, document_type.split('x'))
    doc_ratio = doc_width / doc_height
    img_ratio = w / h

    ratio_diff = max(doc_ratio, img_ratio) / min(doc_ratio, img_ratio)
    #print("Aspect Ratio Difference:", ratio_diff)

    return ratio_diff > 1.1


import cv2
import math
import numpy as np

def detect_skewness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[0:2]
    
    # Invert the colors of our image
    cv2.bitwise_not(gray, gray)
    
    # Hough transform:
    minLineLength = width / 2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    
    # Check if any lines were detected
    if lines is None:
        return False  
    
    # Calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)
    
    for line in lines:
        angle += math.atan2(line[0][3] - line[0][1], line[0][2] - line[0][0])
    
    angle /= nb_lines
    # I set 15 degrees, on my opinion >15 is not good 
    if abs(angle * 180.0 / np.pi) > 10:
        return True, angle * 180.0 / np.pi
    print( angle * 180.0 / np.pi)
    return False

def calculate_overall_score(img, document_type=None):
    is_blurry = detect_blur(img)
    is_overexposed = detect_over_exposure(img)
    is_aspect_ratio_incorrect = detect_aspect_ratio_incorrect(img, document_type)
    is_skewed = detect_skewness(img)
    ocr_confidence = get_ocr_confidence(img)

    print("is_blurry",is_blurry)
    print("is_overexposed",is_overexposed)
    print("is_aspect_ratio_incorrect",is_aspect_ratio_incorrect)
    print("is_skewed",is_skewed)
    print("ocr_confidence",ocr_confidence)

    overall_score = 0
    overall_score += (0 if is_blurry else 1) * 20
    overall_score += (0 if is_overexposed else 1) * 20
    overall_score += (0 if is_aspect_ratio_incorrect else 1) * 20
    overall_score += (0 if is_skewed else 1) * 20
    overall_score += ocr_confidence * 0.2

    print("Overall Score:", overall_score)
    return overall_score
