from dmxpy import DmxPy


if __name__ == '__main__':
	dmx = DmxPy.DmxPy(serial_port='COM3', baud_rate=9600, default_level=0)

	dmx.render()
	dmx.set_channel(1, 128)
	dmx.set_channel(2, 128)
	dmx.set_channel(3, 128)
	dmx.set_channel(4, 128)

	dmx.blackout()
	dmx.render()