from collections import deque

import cv2
import copy
import imutils
from imutils.video import VideoStream
import time
import numpy as np


class AverageBuffer:
	def __init__(self, maxlen):
		self.buffer = deque(maxlen=maxlen)
		self.shape = None

	def apply(self, frame):
		self.shape = frame.shape
		self.buffer.append(frame)

	def get_frame(self):
		mean_frame = np.zeros(self.shape, dtype='float32')
		for item in self.buffer:
			mean_frame += item
		mean_frame /= len(self.buffer)
		return mean_frame.astype('uint8')


class WeightedAverageBuffer(AverageBuffer):
	def get_frame(self):
		mean_frame = np.zeros(self.shape, dtype='float32')
		i = 0
		for item in self.buffer:
			i += 4
			mean_frame += item * i
		mean_frame /= (i * (i + 1)) / 8.0
		return mean_frame.astype('uint8')


# webcam = cv2.VideoCapture(1)

# webcam = VideoStream(src=2).start()
webcam = cv2.VideoCapture("X:\\Silchester Players\\Aladdin\\Footage\\front of tabs only.mp4")
time.sleep(1)
webcam.set(1, 875)

# Background subtraction method (KNN, MOG2)
algo = "KNN"

if algo == 'MOG2':
	backSub = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
else:
	backSub = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False)

weighted_buffer = AverageBuffer(5)

while True:
	start = time.time()
	ret, frame = webcam.read()
	h, w = frame.shape[:2]
	if frame is None:
		break

	# grey = cv2.split(frame)[0]
	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# grey = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[2]

	edges = cv2.Canny(grey, 120, 200)

	# Otsu's thresholding after Gaussian filtering
	blur = cv2.GaussianBlur(edges, (5, 5), 0)
	ret3, fgThres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	blob = fgThres

	frame_f32 = fgThres.astype('float32')
	weighted_buffer.apply(frame_f32)

	blob = backSub.apply(weighted_buffer.get_frame())

	thresh = cv2.adaptiveThreshold(blob, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 17)
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

	# Using morph close to get lines outside the drawing
	remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

	cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	mask = np.zeros(grey.shape, np.uint8)

	boxes_mask = np.zeros(grey.shape, np.uint8)
	boxes_colour = np.zeros(frame.shape, np.uint8)
	height, width = grey.shape
	min_x, min_y = width, height
	max_x = max_y = 0
	rect_num = 0
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		min_x, max_x = min(x, min_x), max(x + w, max_x)
		min_y, max_y = min(y, min_y), max(y + h, max_y)
		if w > 150 and h > 150:
			cv2.rectangle(boxes_mask, (x, y), (x + w, y + h), (255, 0, 0), -1)
			cv2.rectangle(boxes_colour, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.putText(boxes_colour, f"{rect_num}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
			rect_num += 1
		cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

	# output = boxes_mask

	output = cv2.bitwise_and(frame, frame, mask=boxes_mask)

	output += boxes_colour

	

	end = time.time()
	cv2.putText(output, f"{int(1 / (end - start))}FPS", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
	cv2.imshow(f"FG Mask", output)

	# quit with q button
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()
