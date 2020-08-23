import cv2
import numpy as np

img = cv2.imread('NlLrx.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 10
maxLineGap = 1
lines = cv2.HoughLinesP(edges,1,np.pi/180,100, minLineLength,maxLineGap)
print(lines)
for line in lines:
	x1,y1,x2,y2=line[0]
	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('img',img)
cv2.imshow("edges",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()