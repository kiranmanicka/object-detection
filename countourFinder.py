# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import imutils
import time
import numpy as np
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)




imageA=None
imageB=None

checked=False

oldrects=[[0,0,0,0]]
rects=[[0,0,0,0]]

while True:
	ret, frame = vs.read()
	# handle the frame from VideoCapture or VideoStream

	imageB=imageA
	imageA=frame
	#360x640x3






	


	if checked==True:
	
	


		# convert the images to grayscale
		grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
		grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
		(score, diff) = compare_ssim(grayA, grayB, full=True)
		diff = (diff * 255).astype("uint8")
		print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
		thresh = cv2.threshold(diff, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		

		

		lower=(255,255,255)
		upper=(255,255,255)

		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# draw all contours
		#what = cv2.drawContours(imageB, contours, -1, (0, 255, 0), 2)
		oldrects=rects
		rects=[[0,0,0,0]]
		for c in contours:
			if cv2.contourArea(c)<=40:
				continue
			x,y,w,h = cv2.boundingRect(c)
			#for r in range(len(rects)):
				#if x+w >rects[r][0] or y+h>rects[r][3]:
					#rects.remove(rects[r])
					#rects.append([x,y,w,h])
			rects.append([x,y,w,h])
		#newrects=[[0,0,0,0]]
		#for r in range(len(rects)):
		#	for t in range(len(oldrects)):
		#		if abs(rects[r][0]-oldrects[t][0])<20 and abs(rects[r][1]-oldrects[t][1])<20 and abs(rects[r][2]-oldrects[t][2])<20 and abs(rects[r][3]-oldrects[t][3])<20:
		#			newrects.append(rects[r])
			
		#rects=newrects


		for t in rects:
			cv2.rectangle(imageB,(t[0],t[1]),(t[0]+t[2],t[1]+t[3]),(0,255,0),2)
			center=(t[0],t[1])
			print(center)
		
		
				



		#mask = cv2.inRange(hsv, lower, upper)
		


	# find contours in the mask and initialize the current
	# (x, y) center of the ball
		
			#cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
			#cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
		cv2.imshow("Original", imageA)
		cv2.imshow("Modified", imageB)
		cv2.imshow("Diff", diff)
		cv2.imshow("Thresh", thresh)
		


	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	checked=True

	if frame is None:
		break
	key= cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()
