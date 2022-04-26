from numba import jit
import numpy as np
from PIL import Image

# constants
centercolor = (0, 0, 0)
color1 = (0, 0, 255)
color2 = (255, 255, 255)

iterations = 50
isize = (1000, 1000)

location = (0, 0)
zoom = 1.0

step = ((4.0/ zoom)/ isize[0], (4.0 / zoom) / isize[1])
corner = (location[0] - (step[0] * isize[0] / 2), location[1] - (step[1] * isize[1] / 2))

@jit(nopython=True)
def ValueAt(c):
	z = 0.0
	for n in range(0, iterations):
		zn = np.add(np.multiply(z, z), c)
		if np.absolute(zn) > 2.0:
			return (iterations - n) / iterations
		z = zn
	return -1.0

def GetColor(v):
	if (v == -1.0):
		return centercolor
	cv1 = color2[0] + (color1[0] - color2[0]) * v
	cv2 = color2[1] + (color1[1] - color2[1]) * v
	cv3 = color2[2] + (color1[2] - color2[2]) * v
	return (int(cv1), int(cv2), int(cv3))

def GetCords(pix):
	pos = np.add(corner, np.multiply(step, pix))
	return complex(pos[0], pos[1])

def GenImageRGB(pixels):
	rowCount = 0
	for y in range(0, isize[1]):
		for x in range(0, isize[0]):
			v = ValueAt(GetCords((x,y)))
			pixels[x, y] = GetColor(v)
		rowCount = rowCount + 1
		if rowCount % 10 == 0:
			print(rowCount, end='|', flush=True)

img = Image.new(mode = "RGB", size = isize)
pixels = img.load()
GenImageRGB(pixels)
img.save("./output.png", "PNG")
