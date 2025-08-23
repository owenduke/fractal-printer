docstring = """
This file contains functions that define a 3D shape (in 4D quaternion space on
 the hyperplane where the real part is 0) by the directed distance between them 
 and a given point.
"""
import numpy as np
import quaternion

def sphere(z, center = np.quaternion(0,0,0,0), radius = 1):
    return np.abs(z - center) - radius

def torus(z, center = np.quaternion(0,0,0,0), axis = np.quaternion(0,1,0,0), major_radius = 1, minor_radius = 0.1):
    r = z - center
    r_parallel = -(r * axis + axis * r) * axis / 2 # projection for quaternions
    d_parallel = np.abs(r_parallel)
    d_perpendicular = np.abs(r - r_parallel)

    return np.sqrt(np.power(d_parallel,2) + np.power(d_perpendicular-major_radius,2)) - minor_radius
