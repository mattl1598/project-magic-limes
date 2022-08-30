import math

import numpy as np
import serial
from scipy.optimize import curve_fit
import time


def pan_func(xy, a, b, c):
	return a*(xy[0]**2) + b*(xy[0]) + c
	# return math.degrees(math.atan(b/xy[0]))/a + c


def tilt_func(xy):
	return 


if __name__ == '__main__':
	com = serial.Serial("COM5", baudrate=2000000, rtscts=False)

	fade = [
		*list(range(128, 255))[::1],
		*list(range(0, 256))[::-1],
		*list(range(0, 127))[::1]
	]

	x = [1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 1]
	y = [227.8, 229.6, 230.2, 232.5, 235.6, 239.4, 243.8, 248.9, 254.6, 260.9, 267.7, 275, 282.7, 290.8, 299.3]
	pan = [1.7, 5, 8.4, 11.6, 14.8, 17.9, 20.9, 23.8, 26.6, 29.2, 31.7, 34.1, 36.3, 38.5, 40.5]
	tilt = [75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75]

	# print(np.linalg.lstsq([x, y], [pan, tilt], rcond=None)[0])
	# A = np.vstack([x, np.ones(len(x))]).T

	# print(np.polyfit(x, np.array(pan), deg=2))
	# args = np.polyfit(x, np.array(pan), deg=2)
	#
	args = curve_fit(tilt_func, x, pan)[0]
	print(args)



	import matplotlib.pyplot as plt

	_ = plt.plot(x, pan, 'o', label='Original data', markersize=10)
	_ = plt.plot(x, [pan_func([i], *args) for i in x], 'r', label='Fitted line')
	_ = plt.legend()
	plt.show()

	# while True:
	# 	deg = int(input("angle"))
	# 	# dmx = int((deg / 109.0)*32767)  # tilt
	# 	dmx = int((deg / 270.0) * 32767)  # pan
	#
	# 	coarse = (dmx >> 8) & 0xff
	# 	fine = dmx & 0xff
	# 	com.write(bytes(f'<{",".join(["001", f"{coarse:03}"])}>', "ascii"))
	# 	com.write(bytes(f'<{",".join(["003", f"{fine:03}"])}>', "ascii"))
	# for i in fade:
	# 	# print(",".join(["001", f"{i:03}"]))
	# 	com.write(bytes(f'<{",".join(["001",f"{i:03}"])}>', "ascii"))
	# 	com.write(bytes(f'<{",".join(["002",f"{i:03}"])}>', "ascii"))
	# 	com.write(bytes(f'<{",".join(["027",f"{i:03}"])}>', "ascii"))
	# 	# time.sleep(0.5)
