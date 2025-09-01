docstring = """
This file contains functions that define a 3D shape (in 4D quaternion space on
 the hyperplane where the real part is 0) by the directed distance between them 
 and a given point.
"""
import numpy as np
import quaternion

def sphere(z, cx = 0, cy = 0, cz = 0, radius = 1):
    center = quaternion.quaternion(0,cx,cy,cz)
    return np.abs(z - center) - radius

def torus(z, cx = 0, cy = 0, cz = 0, ax = 1, ay = 0, az = 0, major_radius = 1, minor_radius = 0.1):
    center = quaternion.quaternion(0,cx,cy,cz)
    axis = quaternion.quaternion(0,ax,ay,az)
    r = z - center
    r_parallel = -(r * axis + axis * r) * axis / 2 # projection for quaternions
    d_parallel = np.abs(r_parallel)
    d_perpendicular = np.abs(r - r_parallel)

    return np.sqrt(np.power(d_parallel,2) + np.power(d_perpendicular-major_radius,2)) - minor_radius

def generalized_mandelbrot(z0, cx=0, cy=0, cz=0, power = 2, iterations = 10, bailout = 10000):
    z = np.copy(z0)
    d = quaternion.quaternion(1,0,0,0)
    c = quaternion.quaternion(0,cx,cy,cz)

    for n in range(iterations):
        d = power * z**(power-1) * d
        z = z**power + c

        if np.abs(z) > bailout:
            break

    return np.abs(z) * np.log(np.abs(z)) / np.abs(d)

functions = {
    "sphere" : sphere,
    "torus" : torus,
    "generalized_mandelbrot": generalized_mandelbrot
 }