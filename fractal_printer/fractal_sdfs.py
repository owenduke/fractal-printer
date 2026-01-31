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
    return z * z.conj()

@d3.sdf3
def quaternion_julia_sdf(c, w=0, n = 2, iterations = 50, bailout_level = 10000**2, offset=0):

    def distance(p):
        # Convert starting points to quaterinons
        z = quaternion.from_float_array(
            np.concatenate(
                (p,w*np.ones((p.shape[:-1]+(1,)))),
                axis=1
            )
        )

        # Define helper arrays
        z2 = mag2(z)                                    # Current mag^2 of tracked point
        zp2 =         np.ones(p.shape[:-1],dtype=float) # Current mag^2 of derivative
        bailout =     np.zeros(p.shape[:-1],dtype=bool) # Mask of points that have crossed the bailout threshold
        new_bailout = np.zeros(p.shape[:-1],dtype=bool) # Mask of points that have crossed the bailout threshold this iteration

        for _ in range(iterations):
            # Update points
            zp2[~bailout] = n*n*z2[~bailout]*zp2[~bailout]
            z[~bailout] = z[~bailout]**n + c
            z2[~bailout] = mag2(z[~bailout])

            # Update mask
            new_bailout = (z2 > bailout_level) & ~bailout
            bailout = bailout | new_bailout

        dist = np.sqrt(z2/zp2)*0.5*np.log(z2)
        return dist - offset

    return distance