import cv2
from tracker import *


#Creamos el objeto de rastreo
tracker = EuclideanDistTracker() #no recone la libreria o no sé que pasa

cap = cv2.VideoCapture("video.mp4")

#Detector
detector_objeto = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    #Extraer region de interes
    roi = frame[340: 720, 500: 800]


    mask = detector_objeto.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:


        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)








    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)
    cv2.imshow("Roi",roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
