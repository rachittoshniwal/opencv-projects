import cv2 as cv
import sys

s = 0

if len(sys.argv)>1:
    s = sys.argv[1]

haar_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

source = cv.VideoCapture(s)
window_width, window_height = int(source.get(3)), int(source.get(4))

playing = True
to_do = 'normal' # by default, the normal camera visual is played, i.e. without any blurring
while playing:
    isok, frame = source.read()
    frame = cv.flip(frame, 1)

    if not isok:
        break

    if to_do == 'blur': # if key 'b' is pressed for blurring
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_rect = haar_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
        for face in face_rect:
            (x, y, w, h) = face
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv.GaussianBlur(face_roi, (29,29), 15)
            frame[y:y+h, x:x+w, :] = face_roi
    
    cv.putText(frame, f"{to_do} mode", (15,15), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)
    cv.putText(frame, "press 'b' for blur mode, 'n' for back to normal mode", (15,window_height - 30), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)
    cv.putText(frame, "press 'esc' or 'q' or 'Q' to quit", (15,window_height - 15), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)
    cv.imshow("video", frame)

    key = cv.waitKey(1)
    if key == 27 or key == ord('q') or key == ord('Q'):
        playing = False

    elif key == ord('b'):
        to_do = 'blur'
    
    elif key == ord('n'):
        to_do = 'normal'

source.release()
cv.destroyAllWindows()
