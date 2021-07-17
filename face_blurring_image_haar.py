import cv2 as cv

haar_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv.imread('./test images/friends1.jpg')
img = cv.resize(img, (640,480))
cv.imshow("original image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
face_rect = haar_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
for face in face_rect:
    (x, y, w, h) = face
    face_roi = img[y:y+h, x:x+w]
    face_roi = cv.GaussianBlur(face_roi, (29,29), 15)
    img[y:y+h, x:x+w, :] = face_roi
    
cv.imshow("blurred image", img)

cv.waitKey(0)