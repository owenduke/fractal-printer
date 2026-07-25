docstring = """
This file contains functions that define a 3D shape (in 4D quaternion space on
 the hyperplane where the real part is 0) by the directed distance between them 
 and a given point.

Each distance function must return the estimated distance to the surface as well
as the gradient of the distance field (oriented outward) for fast dual contouring.


"""
import threading
import numpy as np
import quaternion
from sdf.sdf import d3

# numpy-quaternion's C extension isn't thread-safe: concurrent calls into it
# (as happens when sdf.generate's ThreadPool workers each evaluate a quaternion
# SDF) corrupt its internal dtype refcounting and crash the interpreter. Every
# quaternion operation below is serialized through this lock so the threaded
# generation path stays crash-free without giving up parallelism for the
# non-quaternion work (marching cubes, array prep) that runs alongside it.
_QUATERNION_LOCK = threading.Lock()


def mag2(z):
    return quaternion.as_float_array(z * z.conj())[...,0]


@d3.sdf3
def quaternion_julia_sdf(cx = 0, cy=0, cz=0, cw=0, slice=0, power = 2, iterations = 50, bailout = 10000**2, offset=0, fudge_factor = 0.9):

    c = quaternion.from_float_array((cx, cy, cz, cw))
    def distance(p):
        with _QUATERNION_LOCK:
            # Convert starting points to quaterinons
            z = quaternion.from_float_array(
                np.concatenate(
                    (p,slice*np.ones((p.shape[:-1]+(1,)))),
                    axis=1
                )
            )

            # Define helper arrays
            z2 = mag2(z)                                    # Current mag^2 of tracked point
            zp2 =         np.ones(p.shape[:-1],dtype=float) # Current mag^2 of derivative
            mask =     np.zeros(p.shape[:-1],dtype=bool)    # Mask of points that have crossed the bailout threshold
            new_mask = np.zeros(p.shape[:-1],dtype=bool)    # Mask of points that have crossed the bailout threshold this iteration

            for _ in range(iterations):
                # Update points
                zp2[~mask] = power*power*np.power(z2[~mask],power-1) * zp2[~mask]
                #zp2 = np.clip(zp2, min=1e-6)
                z[~mask] = np.power(z[~mask],power) + c
                z2[~mask] = mag2(z[~mask])

                # Update mask
                new_mask = (z2 > bailout) & ~mask
                mask = mask | new_mask


            dist = np.sqrt(z2/zp2)*np.log(z2)/(2*power)
            dist[~mask] = -1
            return (dist - offset) * fudge_factor

    return distance

def polynomial_update(coefficients):
    
    C = quaternion.from_float_array(coefficients)
    def update(z):
        z_1 = np.zeros_like(z)
        zp_1 = np.zeros_like(z)
        for i, c in enumerate(C):
            if i == 0:
                z_1 = z_1 + c
            elif i == 1:
                z_1 = z_1 + z * c
                zp_1 = zp_1 + c
            else:
                z_1 = z_1 + np.power(z, i) * c
                zp_1 = zp_1 + np.power(z, i-1) * i * c
                
        return z_1, zp_1
    return update

@d3.sdf3
def general_julia_sdf(update, slice=0, power = 2, iterations = 50, bailout = 10000**2, offset=0, fudge_factor = 0.9):

    def distance(p):
        with _QUATERNION_LOCK:
            # Convert starting points to quaterinons
            z = quaternion.from_float_array(
                np.concatenate(
                    (p,slice*np.ones((p.shape[:-1]+(1,)))),
                    axis=1
                )
            )
            zp = quaternion.from_float_array([[1,0,0,0]]*p.shape[0])

            # Define helper arrays
            z2 = mag2(z)                                    # Current mag^2 of tracked point
            # zp2 =      np.ones(p.shape[:-1],dtype=float)    # Current mag^2 of derivative
            mask =     np.zeros(p.shape[:-1],dtype=bool)    # Mask of points that have crossed the bailout threshold
            new_mask = np.zeros(p.shape[:-1],dtype=bool)    # Mask of points that have crossed the bailout threshold this iteration

            for _ in range(iterations):
                # Update points
                z_1, zp_1 = update(z[~mask])
                zp[~mask] = zp_1 * zp[~mask]
                #zp2 = np.clip(zp2, min=1e-6)
                z[~mask] = z_1
                z2[~mask] = mag2(z[~mask])

                # Update mask
                new_mask = (z2 > bailout) & ~mask
                mask = mask | new_mask

            zp2 = mag2(zp)
            dist = np.sqrt(z2/zp2)*np.log(z2)/(2*power)
            dist[~mask] = -1
            return (dist - offset) * fudge_factor

    return distance

def polynomial_julia_sdf(coefficients, **kwargs):
    update = polynomial_update(coefficients)
    return general_julia_sdf(update=update,**kwargs)