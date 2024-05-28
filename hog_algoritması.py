import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

# Resmi yükle
image = cv2.imread("../resources/images/ag.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# HOG özelliklerini hesapla
fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# HOG görüntüsünü normalize et
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

cv2.imwrite("../resources/saved/original_saved_image.jpg", image)
cv2.imwrite("../resources/saved/hog_saved_image.jpg", hog_image_rescaled)

# Görüntüleri göster
cv2.imshow("Original Image", image)
cv2.imshow("HOG Image", hog_image_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
