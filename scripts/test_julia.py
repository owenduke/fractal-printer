from fractal_printer import fractal_sdfs as df
import quaternion as qu
import matplotlib.pyplot as plt
import numpy as np
from sdf.sdf.d3 import sphere, torus, slab
from importlib import reload



def main():
    julia_sdf = df.generalized_mandelbrot(0.4,0.2,0.5, iterations=50, offset=0.01)

    # Export a 3d rendering
    radius = 1.2
    julia_sdf.save("test_julia.stl",bounds = ([-radius]*3,[radius]*3), samples=2**26)

if __name__ == "__main__":
    main()