from turtle import pos
from numba import jit
import numpy as np
from PIL import Image
import time

# constants
centercolor = (0, 0, 0)
color1 = (0, 0, 255)
color2 = (255, 255, 255)

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
			return (iterations - n) / iterations
		z = zn
	return -1.0

@jit(nopython=True)
def GetColor(v):
	if (v == -1.0):
		return centercolor
	cv1 = color2[0] + (color1[0] - color2[0]) * v
	cv2 = color2[1] + (color1[1] - color2[1]) * v
	cv3 = color2[2] + (color1[2] - color2[2]) * v
	return (int(cv1), int(cv2), int(cv3))

@jit(nopython=True)
def GetCords(pix):
	return complex(corner[0] + step[0] * pix[0], corner[1] + step[1] * pix[1])

@jit(forceobj=True, parallel=True)
def GenImageRGB(pixels):
	for y in range(0, isize[0]):
		for x in range(0, isize[1]):
			v = ValueAt(GetCords((x,y)))
			pixels[x, y] = GetColor(v)

start = time.time()
img = Image.new(mode = "RGB", size = isize)
pixels = img.load()
GenImageRGB(pixels)
end = time.time()

print("Elapsed = %s" % (end - start))
print("Cords: " + str(location[0]) + ", " + str(location[1]))
print("Zoom: " + str(zoom))

img.save("./output.png", "PNG")