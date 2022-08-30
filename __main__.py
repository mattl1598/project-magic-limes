import statistics
from collections import deque
import platform
import distinctipy
from itertools import permutations
import cv2
import time
import numpy as np
import math

import serial


def send_dmx(com, addr, val):
	com.write(bytes(f'<{",".join([f"{addr:03}",f"{val:03}"])}>', "ascii"))


def on_mouse(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		# draw circle here (etc...)
		print('x = %d, y = %d' % (x, y))


def x_to_angle(x, max_angle):
	return -(0.048*x)+24.213 + max_angle/2


def aim_light(deg, com):
	dmx = int((deg / 270.0) * 32767)  # pan
	coarse = (dmx >> 8) & 0xff
	fine = dmx & 0xff
	send_dmx(com, 1, coarse)
	send_dmx(com, 3, fine)


def least_movement(last_pos, new_pos):
	movements = []
	perms = list(permutations(new_pos))
	for i in range(len(perms)):
		movement = 0
		for j in range(len(perms[i])):
			x_change = int(last_pos[j][0] - int(perms[i][j][0]))
			y_change = int(last_pos[j][1]) - int(perms[i][j][1])
			movement += int(math.sqrt(abs(x_change**2 + y_change**2)))
		movements.append(movement)

	return perms[np.argmin(movements)]


class AverageSpot:
	def __init__(self, maxlen):
		self.cx = deque(maxlen=maxlen)
		self.cy = deque(maxlen=maxlen)
		self.r = deque(maxlen=maxlen)
		self.b = deque(maxlen=maxlen)

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

		return tuple((x, y, r, b))

	def get_last_pos(self):
		x = self.cx[-1]
		y = self.cy[-1]
		r = self.r[-1]
		b = self.b[-1]

		return tuple((x, y, r, b))

	def get_length(self):
		return len(self.cx)

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


if platform.system() != "Windows":
	webcam = cv2.VideoCapture("/dev/video0")
	time.sleep(1)
else:
	# webcam = cv2.VideoCapture("X:\\Silchester Players\\Aladdin\\Footage\\evening full.mp4")
	webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
	time.sleep(1)
	# webcam.set(1, 875)
	# webcam.set(1, 10000)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
ret, frame = webcam.read()

# Params

# Background subtraction method (KNN, MOG2)
algo = "MOG2"
spots = 1
spot_buffer_size = 10
com = serial.Serial("COM5", baudrate=2000000, rtscts=False)

if algo == 'MOG2':
	backSub = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
else:
	backSub = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False)

weighted_buffer = AverageBuffer(15)
spots_buffers = [AverageSpot(spot_buffer_size) for i in range(spots)]

dots = np.zeros(frame.shape, np.uint8)
h, w = frame.shape[:2]
x = np.linspace(0, w, int(w / 10))
y = np.linspace(0, h, int(h / 10))
X, Y = np.meshgrid(x, y)

positions = np.column_stack([X.ravel(), Y.ravel()]).astype(int)

for (x, y) in positions:
	dots[y - 1, x - 1] = [255, 255, 255]

spot_colours = (np.array(distinctipy.get_colors(spots))*255).astype(int).tolist()
# print(spot_colours)
last_centers = None
last_labels = None

gpu = False

if gpu:
	cv = cv2.cuda
else:
	cv = cv2

while True:
	start = time.time()
	ret, frame = webcam.read()
	if not ret:
		break
	h, w = frame.shape[:2]

	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	edges = cv2.Canny(grey, 120, 200)

	# Otsu's thresholding after Gaussian filtering
	blur = cv2.GaussianBlur(edges, (5, 5), 0)
	ret3, fgThres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

	for c in cnts:
		hull = cv2.convexHull(c)
		cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

	# Using morph close to get lines outside the drawing
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

	masked_dots = cv2.bitwise_and(dots, dots, mask=mask)

	points = np.argwhere(cv2.cvtColor(masked_dots, cv2.COLOR_BGR2GRAY)).astype(np.float32)
	points = np.array([point[::-1] for point in points])

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	if points.any():
		ret, label, center = cv2.kmeans(
			points,
			spots,
			last_labels,
			criteria,
			100,
			cv2.KMEANS_RANDOM_CENTERS
		)
	else:
		center = None

	if center is not None:
		centers = []
		new_pos = None
		for point in center:
			centers.append(tuple(point.astype(np.uint32)))
		if spots_buffers[0].get_length():
			last_pos = [spot.get_last_pos()[:2] for spot in spots_buffers]
			centers = least_movement(last_pos, centers)
			# centers = [centers[i] for i in center_indexes]
			# radii = []
		for i in range(len(centers)):
			spots_buffers[i].update(*centers[i], 0, 0)
			if spots == 1:
				aim_light(x_to_angle(spots_buffers[i].get_pos()[0], 540), com)
			cv2.circle(frame, spots_buffers[i].get_pos()[:2], 10, spot_colours[i], -1)

	output = frame

	end = time.time()
	cv2.putText(output, f"{int(1 / (end - start))}FPS", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
	cv2.imshow(f"Window", output)

	# quit with q button
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	cv2.setMouseCallback('Window', on_mouse)

webcam.release()
cv2.destroyAllWindows()
