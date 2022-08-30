import cv2
import platform
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import serial

index = 0

pt = [(270, 19,), (275, 30,), (265, 14,), (280, 25,), (260, 34,)]
xy = []

pargs, targs = [], []


def plane_fit(points):
	"""
	p, n = planeFit(points)

	Given an array, points, of shape (d,...)
	representing points in d-dimensional space,
	fit a d-dimensional plane to the points.
	Return a point, p, on the plane (the point-cloud centroid),
	and the normal, n.
	"""
	from numpy.linalg import svd
	# points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	ctr = points.mean(axis=1)
	x = points - ctr[:,np.newaxis]
	M = np.dot(x, x.T) # Could also use np.cov(x) here.
	norm = svd(M)[0][:, -1]
	d = -(ctr[0]*norm[0] + ctr[1]*norm[1] + ctr[2]*norm[2])
	args = list([*norm, d])
	return args


def pan_func(xy, a, b, c):
	return a*(xy[0]**2) + b*(xy[0]) + c


def tilt_func(xy, a, b, c, d):
	x, y = xy
	return (d + a*x + b*y)/(-c)


def on_mouse(event, x, y, flags, param):
	global index, pargs, targs

	if event == cv2.EVENT_LBUTTONDOWN:
		# draw circle here (etc...)
		print('x = %d, y = %d' % (x, y))
		print(pan_func([x, y], *pargs))
		print(tilt_func([x, y], *targs))
		send_dmx(com, 1, *angle_to_dmx("pan", pan_func([x,y], *pargs)))
		send_dmx(com, 2, *angle_to_dmx("tilt", tilt_func([x,y], *targs)))


def angle_to_dmx(axis, deg):
	if axis == "pan":
		dmx = int((deg / 270.0) * 32767)  # pan
	else:
		dmx = int((deg / 109.0)*32767)  # tilt

	coarse = (dmx >> 8) & 0xff
	fine = dmx & 0xff

	return coarse, fine


def send_dmx(com, addr, coarse, fine=None):
	com.write(bytes(f'<{",".join([f"{addr:03}", f"{coarse:03}"])}>', "ascii"))
	if fine is not None:
		com.write(bytes(f'<{",".join([f"{addr+2:03}", f"{fine:03}"])}>', "ascii"))
	time.sleep(0.1)


# if platform.system() != "Windows":
# 	webcam = cv2.VideoCapture("/dev/video0")
# 	time.sleep(1)
# else:
# 	# webcam = cv2.VideoCapture("X:\\Silchester Players\\Aladdin\\Footage\\evening full.mp4")
# 	webcam = cv2.VideoCapture(1)
# 	time.sleep(1)
# 	# webcam.set(1, 875)
# 	# webcam.set(1, 10000)

start = time.time()
webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print((one := time.time())-start)
com = serial.Serial("COM5", baudrate=2000000, rtscts=False)
print((two := time.time()) - one)

pan = 270
tilt = 19

send_dmx(com, 1, *angle_to_dmx("pan", pan))
send_dmx(com, 2, *angle_to_dmx("tilt", tilt))
send_dmx(com, 26, 20)
send_dmx(com, 25, 255)
send_dmx(com, 27, 255)
time.sleep(2)

print((three := time.time()) - two)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

print((four := time.time()) - three)
time.sleep(1)
ret, frame = webcam.read()

# index = 0
while True:
	start = time.time()
	ret, frame = webcam.read()
	initial_frame = frame
	if not ret:
		break
	h, w = frame.shape[:2]

	if index < len(pt):
		time.sleep(1)
		ret, frame = webcam.read()
		initial_frame = frame
		# cv2.imshow(f"Window", initial_frame)
		time.sleep(1)
		frame = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HLS))[1]
		frame = cv2.GaussianBlur(frame, (5, 5), 0)
		ret3, frame = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
		frame = cv2.erode(frame, None, iterations=2)
		frame = cv2.dilate(frame, None, iterations=4)

		non_zeros = cv2.findNonZero(frame)
		if non_zeros is not None:
			center = (sum(non_zeros) / len(non_zeros)).astype("uint32")[0]
			# color = (0, 255, 0,)
			# cv2.circle(initial_frame, center, 50, color, 3)

		xy.append((center[0], center[1],))
		# cv2.imshow(f"Window", initial_frame)
		print(xy)
		index += 1
		if index == len(pt):
			pargs = np.polyfit([x for x, y in xy], [p for p, t in pt], deg=2)
			# targs = curve_fit(tilt_func, xy, [t for p, t in pt])[0]
			# data = np.array([[xy[i][0], xy[i][1], pt[i][1]] for i in range(len(xy))])
			data = np.array([[x for x, y in xy], [y for x, y in xy], [t for p, t in pt]])

			targs = plane_fit(data)

			print(pargs)
			print(targs)

			index += 1
		elif index < len(pt):
			send_dmx(com, 1, *angle_to_dmx("pan", pt[index][0]))
			send_dmx(com, 2, *angle_to_dmx("tilt", pt[index][1]))
			time.sleep(2)
	# print(index)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if cv2.waitKey(1) & 0xFF == ord('i'):
		deg = int(input("angle"))
		send_dmx(com, 1, *angle_to_dmx("pan", deg))

	cv2.imshow(f"Window", initial_frame)
	cv2.setMouseCallback('Window', on_mouse)

webcam.release()
cv2.destroyAllWindows()
