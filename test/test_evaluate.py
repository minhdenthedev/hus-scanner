import cv2 as cv
import os
from tqdm import tqdm
import asyncio
from src.evaluate.evaluate import calculate_overall_score

images_path = '.\\test_images\\unfiltered_pngs'
corner_path = '.\\test_images\\corner_detection_v2'
warped_path = '.\\test_images\\warped'

list_images = os.listdir(warped_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(warped_path, filename))
        
        print("Test for", filename, ":")
        score = calculate_overall_score(image)
        print(f"Overall score for {filename}: {score}\n")
