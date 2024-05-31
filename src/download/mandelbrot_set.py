from PIL import Image
import numba as nb
import numpy as np
import time
import os

PRECISION = 500
INFINITY = 2

# Image size (pixels)
WIDTH = 600
HEIGHT = int(WIDTH / 1.5)

# constants for the complex plain
# the real axis (or x axis)
RE_START = -2
RE_END = 1
# "imaginary" or lateral axis (or y axis)
IM_START = -1
IM_END = 1

CORES = os.cpu_count()


@nb.njit(fastmath=True, nogil=True)
def get_iterations(c):
    z = complex(0, 0)
    for i in range(PRECISION):
        z = z * z + c  # Zn+1 = Zn^2 + C
        # te absolute value of a complex numbers is defined as real_component ^ 2 + imaginary_component ^ 2
        if abs(z) > INFINITY:
            return i
    return PRECISION


@nb.njit
def point2D(x, y):
    return complex(
            RE_START + (x / WIDTH) * (RE_END - RE_START),
            IM_START + (y / HEIGHT) * (IM_END - IM_START),
        )


@nb.njit(parallel=True)
def mandelbrot_set():
    array = np.zeros((HEIGHT, WIDTH, 3))
    for start in nb.prange(CORES):
        for x in range(start, WIDTH, CORES):
            for y in range(HEIGHT):
                c = point2D(x, y)
                m = get_iterations(c)
                if m < PRECISION:
                    array[y, x] = (
                        int(m + ((m * 20) % 156)),
                        int(100 + ((m * 20) % 156)),
                        int(255),
                    )
                else:
                    array[y, x][:] = 0
    return array


if __name__ == '__main__':
    print("compiling...")
    mandelbrot_set()
    print(f"making a mandelbrot set of size {WIDTH}x{HEIGHT}")
    now = time.time()
    array = mandelbrot_set()
    print(f"it took {time.time() - now}s")

    image = Image.fromarray(np.uint8(array), "RGB")
    image.show()
    # image.save()
