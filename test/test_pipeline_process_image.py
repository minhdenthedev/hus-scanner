import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
# import pyheif
from PIL import Image
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




# # Tạo thư mục để lưu ảnh .jpg đã chuyển đổi
# output_jpg_path = 'test_images/png_images'
# os.makedirs(output_jpg_path, exist_ok=True)
# data_path = 'test_images/new_HEIC_images'

# # Liệt kê các tệp ảnh HEIC trong thư mục
# image_files = [f for f in os.listdir(data_path) if f.endswith('.HEIC')]

# # Đọc từng ảnh HEIC, chuyển đổi sang JPG và lưu lại
# for image_file in image_files:
#     image_path = os.path.join(data_path, image_file)

#     # Đọc ảnh HEIC
#     heif_file = pyheif.read(image_path)
#     image = Image.frombytes(
#         heif_file.mode,
#         heif_file.size,
#         heif_file.data,
#         "raw",
#         heif_file.mode,
#         heif_file.stride,
#     )

#     # Chuyển ảnh thành định dạng RGB nếu ảnh gốc là RGBA hoặc khác
#     image = image.convert("RGB")

#     # Đường dẫn mới lưu ảnh JPEG
#     output_jpg_file = os.path.join(output_jpg_path, f"{os.path.splitext(image_file)[0]}.png")

#     # Lưu ảnh dưới dạng JPEG
#     image.save(output_jpg_file, "JPEG")

#     # In ra đường dẫn của ảnh đã chuyển đổi
#     print(f"Đã chuyển đổi ảnh {image_file} thành {output_jpg_file}")

#     # Hiển thị ảnh JPEG mới (tuỳ chọn)
#     img_cv = cv.imread(output_jpg_file)
#     plt.imshow(cv.cvtColor(img_cv, cv.COLOR_BGR2RGB))  # Chuyển đổi từ BGR sang RGB để hiển thị đúng màu
#     plt.title(f"Ảnh {image_file} đã chuyển đổi")
#     plt.axis('off')
#     plt.show()

