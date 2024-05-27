#kutuphaneleri import etme
import face_recognition as fr
import cv2
import numpy as np
import os
from datetime import datetime

#face encoding
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('yoklama_listesi.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            print(f"{name} {dt_string}")
            f.writelines(f'\n{name},{dt_string}')

def main():
    #sinif listesini olusturma
    path = "kaynaklar/sinif_listesi"
    images = []     # tüm resimleri içeren liste
    classNames = []    # resimlere karşılık gelen sınıfları içeren liste
    myList = os.listdir(path) # dizindeki dosya veya klasorleri bir listeye atar

    print("Toplam Sinif Mevcudu: ",len(myList))

    for cls in myList:
            curImg = cv2.imread(f'{path}/{cls}')
            images.append(curImg)
            classNames.append(os.path.splitext(cls)[0])

    print(classNames)
    
    encodeListKnown = findEncodings(images)
    print("Encoding Complete")
    print("Starting OpenCV...")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame_show = cv2.resize(frame, (480, 640))
        frameS = cv2.resize(frame, (0, 0),None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

        facesCurFrame = fr.face_locations(frameS)
        encodesCurFrame = fr.face_encodings(frameS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = fr.compare_faces(encodeListKnown, encodeFace)
            faceDis = fr.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                markAttendance(name)

        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == '__main__':
    main()