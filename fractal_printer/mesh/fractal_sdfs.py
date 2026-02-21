docstring = """
This file contains functions that define a 3D shape (in 4D quaternion space on
 the hyperplane where the real part is 0) by the directed distance between them 
 and a given point.

Each distance function must return the estimated distance to the surface as well
as the gradient of the distance field (oriented outward) for fast dual contouring.


"""
import numpy as np
import quaternion
from sdf.sdf import d3


def mag2(z):
    return quaternion.as_float_array(z * z.conj())[...,0]


@d3.sdf3
def quaternion_julia_sdf(cx = 0, cy=0, cz=0, cw=0, slice=0, power = 2, iterations = 50, bailout = 10000**2, offset=0, fudge_factor = 0.9):

    c = quaternion.from_float_array((cx, cy, cz, cw))
    def distance(p):
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
        mask =     np.zeros(p.shape[:-1],dtype=bool) # Mask of points that have crossed the bailout threshold
        new_mask = np.zeros(p.shape[:-1],dtype=bool) # Mask of points that have crossed the bailout threshold this iteration

        for _ in range(iterations):
            # Update points
            zp2[~mask] = power*power*np.power(z2[~mask],power-1)*zp2[~mask]
            z[~mask] = z[~mask]**power + c
            z2[~mask] = mag2(z[~mask])

            # Update mask
            new_mask = (z2 > bailout) & ~mask
            mask = mask | new_mask

        dist = np.sqrt(z2/zp2)*0.25*np.log(z2)
        return (dist - offset) * fudge_factor

    return distance