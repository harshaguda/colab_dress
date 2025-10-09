import cv2
import sys

cam_id = sys.argv[1] if len(sys.argv) > 1 else 0
cap=cv2.VideoCapture(int(cam_id))
# print(cap.isOpened(), cam_id)
ret,frame=cap.read()
while ret:
    ret,frame=cap.read()
    if ret:
        cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()