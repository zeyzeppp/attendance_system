import cv2
import numpy as np
import face_recognition as fr
import os

imgCr = fr.load_image_file("../resources/images/img.jpg")
imgCr = cv2.cvtColor(imgCr, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgCr)[0]
print(faceLoc)
encodeCr = fr.face_encodings(imgCr)[0]
cv2.rectangle(imgCr, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)  # top, right, bottom, left

cv2.imshow("CR7", imgCr)
cv2.waitKey(0)
