import cv2
import numpy as np

image = cv2.imread('1.jpg')
cv2.namedWindow('frame',0)
cv2.resizeWindow('frame',640,480)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(src=image,ddepth=cv2.CV_16S,dx=1,dy=0)
y = cv2.Sobel(src=image,ddepth=cv2.CV_16S,dx=0,dy=1)

absX = cv2.convertScaleAbs(x)   # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image=image,contours=contours,contourIdx=-1,color=255,thickness=3)
cv2.imshow('frame', dst)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()