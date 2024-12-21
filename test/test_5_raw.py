# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt

# from src.binarizer.binarizer import Binarizer
# from src.binarizer.remove_shadow import RemoveShadow
# from src.pipeline import Pipeline
# from src.corner_detector.corner_pipeline import CornerPipeline
# from src.warping.warping import Warping
# from src.flatten_image.flatten_image import FlattenImage
# from src.find_contours.find_contours import FindContours

# from src.utils import find_top_2_largest_distances, fill_image_verticles
# import os
# from tqdm import tqdm

# images_path = '.\\example_input'
# corner_path = '.\\test_images\\corner_detection_v2'
# warped_path = '.\\test_images\\warped'
# flatten_path = '.\\test_images\\flatten'
# boundary_path = '.\\test_images\\boundary'


# list_images = os.listdir(images_path)

# if __name__ == '__main__':
#     for filename in tqdm(list_images):
#         image = cv.imread(os.path.join(images_path, filename))
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         height, width = gray.shape
#         center_point = (height/2, width/2)

#         pipeline = Pipeline(stages=[
#             RemoveShadow(),
#             Binarizer()
#         ])

#         binary = pipeline.execute(gray)

#         corners = CornerPipeline(version="v2").execute(gray)
#         # points = [(0, 0), (5, 5), (0, 5), (5, 0), (2, 3)]
#         top_2_distances = find_top_2_largest_distances(corners, width, height)
#         print()
#         verticles = []
#         print("file: ",filename.split("_")[0])
#         for (point1, point2), distance in top_2_distances:
#             verticles.append(point1)
#             verticles.append(point2)
#             print(f"Cặp điểm: {point1}, {point2} - Khoảng cách: {distance:.2f}")


#         for point in verticles:
#             cv.circle(image, point, 20, (0, 255, 0), 20)


#         # Fill verticle if need 
#         if len(verticles) != 4:
#             print(f"Invalid number of vertices ({len(verticles)}) for warping in {filename}.Starting auto fill:")
#             verticles_dict=fill_image_verticles(center_point,verticles)
#             verticles =  [value for key, value in verticles_dict.items()]
#             print('vertices:', verticles)
#         # Points to approx 
#         approx = np.array(verticles, dtype=np.float32).reshape((-1, 1, 2))


#         # # Warping work
#         # warping_only = Pipeline(stages=[
#         #     Warping(approx)
#         # ])
#         # warped_image = warping_only.execute(image)
        
#         # Assuming the pipeline and all necessary functions are defined
#         flatten_only = Pipeline(stages=[
#             FlattenImage(approx),  # Assuming 'approx' is already defined
#         ])

#         # Apply the flattening process
#         warping_img = flatten_only.execute(binary)

#         # Contour detection pipeline
#         contour_only = Pipeline(stages=[
#             FindContours()  # Instantiate the FindContours class
#         ])

#         try:
#             # Execute the contour detection pipeline
#             contours = contour_only.execute(warping_img)

#             # Check if contours were found
#             if contours:
#                 # Create a black background image to draw contours on
#                 boundary_image = np.zeros_like(warping_img, dtype=np.uint8)  # Black background

#                 # Create a kernel for dilation and erosion
#                 kernel = np.ones((5, 5), np.uint8)  # 5x5 square kernel

#                 # Dilate the contours to make them thicker
#                 dilated_image = cv.dilate(boundary_image, kernel, iterations=10)

#                 # Erode the contours to reduce thickness and prevent them from merging
#                 eroded_image = cv.erode(dilated_image, kernel, iterations=30)

#                 # Draw the contours on the boundary image (after dilation and erosion)
#                 cv.drawContours(eroded_image, contours, -1, (255), 14)  # White color, thickness 10

#                 # Construct the filename and save the image with drawn contours
#                 output_filename = filename.split("_")[0] + "_bounded.png"
#                 cv.imwrite(os.path.join(boundary_path, output_filename), eroded_image)
#                 print(f"Saved image with contours: {output_filename}")
#             else:
#                 print("No contours found in the binary image.")

#         except Exception as e:
#             print(f"Error processing the image: {e}")






import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.pipeline import Pipeline
from src.corner_detector.corner_pipeline import CornerPipeline
from src.warping.warping import Warping
from src.flatten_image.flatten_image import FlattenImage
from src.find_contours.find_contours import FindContours

from src.utils import find_top_2_largest_distances, fill_image_verticles
import os
from tqdm import tqdm

images_path = '.\\example_input'
corner_path = '.\\test_images\\corner_detection_v2'
warped_path = '.\\test_images\\warped'
flatten_path = '.\\test_images\\flatten'
boundary_path = '.\\test_images\\boundary'


list_images = os.listdir(images_path)

if __name__ == '__main__':
    for filename in tqdm(list_images):
        image = cv.imread(os.path.join(images_path, filename))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        center_point = (height/2, width/2)

        pipeline = Pipeline(stages=[
            RemoveShadow(),
            Binarizer()
        ])

        binary = pipeline.execute(gray)

        corners = CornerPipeline(version="v2").execute(gray)
        # points = [(0, 0), (5, 5), (0, 5), (5, 0), (2, 3)]
        top_2_distances = find_top_2_largest_distances(corners, width, height)
        print()
        verticles = []
        print("file: ",filename.split("_")[0])
        for (point1, point2), distance in top_2_distances:
            verticles.append(point1)
            verticles.append(point2)
            print(f"Cặp điểm: {point1}, {point2} - Khoảng cách: {distance:.2f}")


        for point in verticles:
            cv.circle(image, point, 20, (0, 255, 0), 20)


        # Fill verticle if need 
        if len(verticles) != 4:
            print(f"Invalid number of vertices ({len(verticles)}) for warping in {filename}.Starting auto fill:")
            verticles_dict=fill_image_verticles(center_point,verticles)
            verticles =  [value for key, value in verticles_dict.items()]
            print('vertices:', verticles)
        # Points to approx 
        approx = np.array(verticles, dtype=np.float32).reshape((-1, 1, 2))


        # # Warping work
        # warping_only = Pipeline(stages=[
        #     Warping(approx)
        # ])
        # warped_image = warping_only.execute(image)
        
        # Assuming the pipeline and all necessary functions are defined
        flatten_only = Pipeline(stages=[
            FlattenImage(approx),  # Assuming 'approx' is already defined
        ])

        # Apply the flattening process
        warping_img = flatten_only.execute(binary)

        # Contour detection pipeline
        contour_only = Pipeline(stages=[
            FindContours()  # Instantiate the FindContours class
        ])
        output_filename = filename.split("_")[0] + "_warped.png"
        cv.imwrite(os.path.join(warped_path, output_filename), warping_img)
        print(f"Saved image with wapred: {output_filename}")

        # try:
        #     # Execute the contour detection pipeline
        #     contours = contour_only.execute(warping_img)

        #     # Check if contours were found
        #     if contours:
        #         # Create a black background image to draw contours on
        #         boundary_image = np.zeros_like(warping_img, dtype=np.uint8)  # Black background

        #         # Draw the contours on the boundary image
        #         cv.drawContours(boundary_image, contours, -1, (255), 23)  # White color, thickness 10

        #         # Construct the filename and save the image with drawn contours
        #         output_filename = filename.split("_")[0] + "_bounded.png"
        #         cv.imwrite(os.path.join(boundary_path, output_filename), boundary_image)
        #         print(f"Saved image with contours: {output_filename}")
        #     else:
        #         print("No contours found in the binary image.")

        # except Exception as e:
        #     print(f"Error processing the image: {e}")