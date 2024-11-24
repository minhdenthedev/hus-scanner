import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("../test_images/IMG_8511.jpg", 0)

dilated_image = cv.dilate(img, np.ones((7, 7), np.uint8))

bg_img = cv.medianBlur(dilated_image, 21)

diff_img = 255 - cv.absdiff(img, bg_img)

norm_img = diff_img.copy()
cv.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

_, thr_img = cv.threshold(norm_img, 230, 0, cv.THRESH_TRUNC)
cv.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

# Apply binary thresholding to create a black-and-white image
_, binary_img = cv.threshold(thr_img, 240, 255, cv.THRESH_BINARY)

# Ensure the output is normalized (optional for binary images, as values are 0 or 255)
cv.normalize(binary_img, binary_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)


fig, axs = plt.subplots(1, 2)
axs[0].imshow(img, cmap="gray")
axs[1].imshow(binary_img, cmap="gray")
plt.show()
