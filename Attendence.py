import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import unittest
from unittest.mock import MagicMock

path = 'imagesDataset'
images = []
classNames = []
myList = os.listdir(path)

for clName in myList:
    curImg = cv2.imread(path+'/'+clName)
    images.append(curImg)
    classNames.append(os.path.splitext(clName)[0])

def imageEncodings(images):
    encodings = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageEncoded = face_recognition.face_encodings(image)[0]
        encodings.append(imageEncoded)
    return encodings

def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        myDataList= f.readlines()
        nameList = []
        for myData in myDataList:
            entry = myData.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now  = datetime.now()
            dstring  = now.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'\n{name}, {dstring}')

class TestFaceRecognition(unittest.TestCase):
    def test_imageEncodings(self):
        mock_encoding = np.random.rand(128) # Generate a random encoding of length 128
        face_recognition.face_encodings = MagicMock(return_value=[mock_encoding])

        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        encodings = imageEncodings(images)

        self.assertEqual(len(encodings), 3)
        for encoding in encodings:
            self.assertEqual(len(encoding), 128)

    def test_markAttendence(self):
        def test_mark_attendance(self):
            tmp_file = 'test_attendance.csv'
            with open(tmp_file, 'w') as f:
                f.write('Alp Arselan, 01/01/2024 12:00:00')

            markAttendence('Mehmet Mustafa', tmp_file)
            with open(tmp_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
                self.assertIn('Mehmet Mustafa', lines[1])

            markAttendence('Alp Arselan', tmp_file)
            with open(tmp_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
            os.remove(tmp_file)

''' if __name__ == '__main__':
    unittest.main()'''

encodingsList = imageEncodings(images)
print("Encoding Complete")

camCaption = cv2.VideoCapture(0)

while True:
    ret, frame = camCaption.read()
    frameSmall = cv2.resize(frame, (0, 0), None,fx=0.25, fy=0.25)
    frameSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(frameSmall)
    encodings = face_recognition.face_encodings(frameSmall , faces)

    for faceLoc , encoding in zip(faces, encodings):
        matches = face_recognition.compare_faces(encodingsList,encoding )
        face_distances = face_recognition.face_distance(encodingsList,encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = classNames[best_match_index]
            y1, x2 , y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame , (x1, y1), (x2, y2), (0, 255, 0) , 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1  , (255, 255, 255) , 2)
            markAttendence(name)
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break






