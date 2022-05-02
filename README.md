# Mandelbrot Set
A quick python implementation of a renderer for the Mandelbrot set. In case you are looking for something that doesn't any preinstalls, is more performant and easier to use and configure you might want to check out our implementation in C (prebuilt binaries for Windows and Linux present) at https://github.com/HiWiSciFi/mandelbrot-c
## Get Started
Since python is..... well..... python -- you sadly have to install the dependencies first.
Use the following commands to install `numpy`, `pillow` and `numba`:<br/>
`pip install numpy`<br/>
`pip install pillow`<br/>
`pip install numba`<br/>
In order to generate an image open up the mbt.py file and edit the constants to your liking (set location, interations and zoom values). Once you're done with that you can simply execute `python mbt.py` in the root directory and see the image being generated (output file: `output.png`).

## Nice images we generated
![img1](./rendered/img1.png)
![img2](./rendered/img2.png)
![img3](./rendered/img3.png)
![img4](./rendered/img4.png)
![img5](./rendered/img5.png)
![img6](./rendered/img6.png)
