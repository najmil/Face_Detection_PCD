import cv2

faceCascade = cv2.CascadeClassifier('D:/pcd/haarcascade_frontalface.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Face Detection by Najmil', img)

    k = cv2.waitKey(1)& 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()