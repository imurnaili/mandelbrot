from numba import jit
import numpy as np
from PIL import Image
import time

# constants
centercolor = (0, 0, 0)
color1 = (0, 0, 225)
color2 = (255, 255, 225)
color3 = (255, 255, 255)

#quality
iterations = 30
#image resolution
isize = (1000, 1000)

#center coordinates
location = (0, 0)
#zoom factor (increase iterations as well)
zoom = 1.0

############
step = ((4.0/ zoom)/ isize[0], (4.0 / zoom) / isize[1])
corner = (location[0] - (step[0] * isize[0] / 2), location[1] - (step[1] * isize[1] / 2))

@jit(nopython=True)
def ValueAt(c):
	z = 0.0
	for n in range(0, iterations):
		zn = np.add(np.multiply(z, z), c)
		if np.absolute(zn) > 2.0:
			return n
		z = zn
	return -1.0

@jit(nopython=True)
def GetColor(v):
	if (v == -1.0):
		return centercolor
	v = (iterations - v) / iterations
	cv1 = color2[0] + (color1[0] - color2[0]) * v
	cv2 = color2[1] + (color1[1] - color2[1]) * v
	cv3 = color2[2] + (color1[2] - color2[2]) * v
	return (int(cv1), int(cv2), int(cv3))

@jit(forceobj=True)
def GetColor_alt1(v):
	if (v == -1.0):
		return centercolor
	if (v == 0):
		return (0, 0, 0)
	options = {
		0 : (66, 30, 15),
		1 : (25, 7, 26),
		2 : (9, 1, 47),
		3 : (4, 4, 73),
		4 : (0, 7, 100),
		5 : (12, 44, 138),
		6 : (24, 82, 177),
		7 : (57, 125, 209),
		8 : (134, 181, 229),
		9 : (211, 236, 248),
		10 : (241, 233, 191),
		11 : (248, 201, 95),
		12 : (255, 170, 0),
		13 : (204, 128, 0),
		14 : (153, 87, 0),
		15 : (106, 52, 3)
	}
	return options[(v) % 16]

@jit(nopython=True)
def GetColor_alt2(v):
	if (v == -1.0):
		return centercolor
	if (v < iterations / 2):
		v = (iterations / 2 - v) / (iterations / 2)
		cv1 = color2[0] + (color1[0] - color2[0]) * v
		cv2 = color2[1] + (color1[1] - color2[1]) * v
		cv3 = color2[2] + (color1[2] - color2[2]) * v
		return (int(cv1), int(cv2), int(cv3))
	else:
		v = (iterations / 2 - (v - iterations / 2)) / (iterations / 2)
		cv1 = color3[0] + (color2[0] - color3[0]) * v
		cv2 = color3[1] + (color2[1] - color3[1]) * v
		cv3 = color3[2] + (color2[2] - color3[2]) * v
		return (int(cv1), int(cv2), int(cv3))
	

@jit(nopython=True)
def GetCoords(pix):
	return complex(corner[0] + step[0] * pix[0], corner[1] + step[1] * pix[1])

@jit(forceobj=True, parallel=True)
def GenImageRGB(pixels):
	for y in range(0, isize[0]):
		for x in range(0, isize[1]):
			v = ValueAt(GetCoords((x,y)))
			pixels[x, y] = GetColor(v)

start = time.time()
img = Image.new(mode = "RGB", size = isize)
pixels = img.load()
GenImageRGB(pixels)
end = time.time()

print("Elapsed = %s" % (end - start))
print("Coords: " + str(location[0]) + ", " + str(location[1]))
print("Zoom: " + str(zoom))
print("Iterations: " + str(iterations))

img.save("./output.png", "PNG")