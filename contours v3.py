import statistics
from collections import deque

import cv2
import copy
import imutils
from imutils.video import VideoStream
import time
import numpy as np


class AverageSpot:
	def __init__(self, maxlen):
		self.cx = deque(maxlen=maxlen)
		self.cy = deque(maxlen=maxlen)
		self.r = deque(maxlen=maxlen*2)
		self.b = deque(maxlen=maxlen*2)

	def update(self, x, y, r, b):
		self.cx.append(x)
		self.cy.append(y)
		self.r.append(r)
		self.b.append(b)

	def get_pos(self):
		x = int(sum(self.cx)/len(self.cx))
		y = int(sum(self.cy)/len(self.cy))
		r = int(sum(self.r)/len(self.r))
		b = sum(self.b)/len(self.b)

		return (x,y,r,b)


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
webcam = cv2.VideoCapture("X:\\Silchester Players\\Aladdin\\Footage\\evening full.mp4")
time.sleep(1)
webcam.set(1, 875)
# webcam.set(1, 10000)

# Background subtraction method (KNN, MOG2)
algo = "KNN"

if algo == 'MOG2':
	backSub = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
else:
	backSub = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False)

weighted_buffer = AverageBuffer(15)

spots = 2
spot_buffer_size = 15
spots_buffers = [AverageSpot(spot_buffer_size) for i in range(spots)]

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

	thresh = cv2.adaptiveThreshold(blob, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 17)
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

	# Using morph close to get lines outside the drawing
	remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

	kernel = np.ones((5, 5), np.uint8)
	remove_horizontal = cv2.erode(remove_horizontal, kernel, iterations=1)

	cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	mask = np.zeros(grey.shape, np.uint8)



	height, width = grey.shape
	min_x, min_y = width, height
	max_x = max_y = 0
	rect_num = 0
	try:
		area_avg = statistics.mean([cv2.contourArea(cv2.convexHull(c)) for c in cnts])
	except:
		area_avg = 0

	for c in cnts:
		hull = cv2.convexHull(c)
		# if cv2.contourArea(hull) > 2*area_avg:
		cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)


	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

	# Using morph close to get lines outside the drawing
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

	dots = np.zeros(frame.shape, np.uint8)
	h, w = frame.shape[:2]
	x = np.linspace(0, w, int(w / 10))
	y = np.linspace(0, h, int(h / 10))

	X, Y = np.meshgrid(x, y)

	positions = np.column_stack([X.ravel(), Y.ravel()]).astype(int)

	for (x, y) in (positions):
		# cv2.circle(dots, (x, y), 1, (255, 0, 0), -1)
		dots[y-1,x-1] = [255,255,255]

	masked_dots = cv2.bitwise_and(dots, dots, mask=mask)

	points = np.argwhere(cv2.cvtColor(masked_dots, cv2.COLOR_BGR2GRAY)).astype(np.float32)
	points = np.array([point[::-1] for point in points])

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	if points.any():
		ret, label, center = cv2.kmeans(points, 2, None, criteria, 100, cv2.KMEANS_PP_CENTERS)
	else:
		center = None

	if center is not None:
		centers = []
		for point in center:
			centers.append(tuple(point.astype(np.uint32)))
		centers.sort(key=lambda x: x[1])
		print(centers)
		cv2.circle(masked_dots, centers[0], 10, (0, 0, 255), -1)
		cv2.circle(masked_dots, centers[1], 10, (0, 255, 0), -1)
		# for c in center:
		# 	print(type(c))q

	# output = cv2.subtract(lights_mask, 255-frame)
	# output.astype(np.uint8)

	# output += boxes_colour

	output = masked_dots

	end = time.time()
	cv2.putText(output, f"{int(1 / (end - start))}FPS", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
	cv2.imshow(f"FG Mask", output)

	# quit with q button
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()
