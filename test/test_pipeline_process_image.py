import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.corner_detector.corner_detector import CornerDetector
from src.warping.warping import Warping
from src.pipeline import Pipeline
from src.utils import detect_corner

# Create a pipeline
pipeline = Pipeline(stages=[
    RemoveShadow(),
    Binarizer(),
    CornerDetector(),
    Warping()
])

# Load biến môi trường từ file .env,setup output folder
load_dotenv()
output_folder = "output"

image_folder_path = Path(os.getenv("IMAGE_FOLDER_PATH"))
for filename in os.listdir(image_folder_path):
    # Load ảnh từ folder để chạy hàng loạt
    image_path = os.path.join(image_folder_path, filename)
    img = cv.imread(image_path, 0)

    
    output_img = pipeline.execute(img)
    # Lưu ảnh vào thư mục output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, "processed_"+filename)
    cv.imwrite(output_path, output_img)
    print(f"Image saved to {output_path}")  # Thông báo vị trí lưu ảnh

