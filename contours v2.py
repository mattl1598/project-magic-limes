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

	lights_mask = np.zeros(frame.shape, np.uint8)
	boxes_mask = np.zeros(grey.shape, np.uint8)
	boxes_colour = np.zeros(frame.shape, np.uint8)
	height, width = grey.shape
	min_x, min_y = width, height
	max_x = max_y = 0
	rect_num = 0
	areas = []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		if w > 150 and h > 150:
			rad = int(np.sqrt(w**2 + h**2)/2)
			center = (int(x+w/2), int(y+h/2))
			min_x, max_x = min(x, min_x), max(x + w, max_x)
			min_y, max_y = min(y, min_y), max(y + h, max_y)
			rect = (x+15, y+15), (x + w-30, y + h-30)
			circ = center, rad
			area = w*h
			areas.append((area, *center, rad))
		hull = cv2.convexHull(c)
		cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
	areas.sort(key=lambda tup: tup[2])
	areas = areas[:len(spots_buffers)]
	areas.sort(key=lambda tup: tup[1])
	for i in range(len(spots_buffers)):
		try:
			spots_buffers[i].update(*areas[i][1:], 1)
		except IndexError:
			try:
				spots_buffers[i].update(*spots_buffers[i].get_pos()[0:-1], 0)
			except ZeroDivisionError:
				spots_buffers[i].update(960, 540, 150, 0)

	for i in range(len(areas)):
		# cv2.rectangle(boxes_mask, *area.get_frame()[1], (255, 0, 0), -1)
		# cv2.rectangle(boxes_colour, *area.get_frame()[1], (255, 0, 0), 2)
		# cv2.putText(boxes_colour, f"{rect_num}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

		# cv2.circle(lights_mask, (areas[i][1], areas[i][2]), areas[i][3], (255, 255, 255), -1)
		circ = spots_buffers[i].get_pos()
		print(circ)
		if circ:
			cv2.circle(lights_mask, (circ[0], circ[1]), circ[2]+20, (circ[3]*255, circ[3]*255, circ[3]*255), -1)

	# output = cv2.subtract(lights_mask, 255-frame)
	# output.astype(np.uint8)

	# output += boxes_colour

	output = mask

	end = time.time()
	cv2.putText(output, f"{int(1 / (end - start))}FPS", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
	cv2.imshow(f"FG Mask", output)

	# quit with q button
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()
