import cv2
import copy
import imutils
from imutils.video import VideoStream
import time

# webcam = cv2.VideoCapture(1)

webcam = VideoStream(src=0).start()
time.sleep(1)


i = 0

while True:
	frame = webcam.read()

	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	median = cv2.medianBlur(frame, 3)

	rect = copy.copy(median)
	cv2.rectangle(rect, (384, 0), (510, 128), (0, 255, 0), 3)

	# Display the resulting frame
	cv2.imshow('raw', frame)
	cv2.imshow('smoothed', median)
	cv2.imshow('rectangle', rect)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.stop()
cv2.destroyAllWindows()