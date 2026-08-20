docstring = """
This file contains functions that define a 3D shape by the directed distance between them 
 and a given point.

Each distance function must return the estimated distance to the surface as well
as the gradient of the distance field (oriented outward) for fast dual contouring.
"""

import threading
import numpy as np
import quaternion
from numba import njit, prange
from sdf import d3

##########################################################
# Flexible pipeline (quaternion module, non-numba, slow) #
##########################################################

# numpy-quaternion's C extension isn't thread-safe: concurrent calls into it
# (as happens when sdf.generate's ThreadPool workers each evaluate a quaternion
# SDF) corrupt its internal dtype refcounting and crash the interpreter.
_QUATERNION_LOCK = threading.Lock()


def mag2(z):
    """
    Compute the squared magnitude of an array of quaternion objects. 
    """
    return quaternion.as_float_array(z * z.conj())[...,0]

@d3.sdf3
def general_julia_sdf(update, 
                      slice: float = 0, power: int = 2, 
                      iterations: int = 50, bailout: float = 10000**2, offset = 0, 
                      interior_epsilon: float = 1e-3, fudge_factor: float = 0.9):
    """
    Generate sdf3 object for a Julia set defined by a generic update map.

    Args:
        update (function): Function (quaternion -> quaternion) defining iterated map. Must be vectorizeable. 
        slice (float, optional): Slice along fourth coordinate in which to define SDF. Defaults to 0.
        power (int, optional): Effective power to use in estimating distance. For polynomial maps, this is the highest
            power in the polynomial. Defaults to 2.
        iterations (int, optional): How many iterations to use to compute the distance estimate. Higher values give 
            more intricate shapes and more accurate distance estimates. Defaults to 50.
        bailout (float, optional): Value at which to stop iterations for diverging points. Defaults to 10000**2.
        offset (int, optional): Offset from the fractal surface at which to define the SDF 0-level. Defaults to 0.
        interior_epsilon (float, optional): Distance to return for points on the interior of the fractal.
            This should be smaller than the meshing grid size to avoid missing chunks. Defaults to 1e-3.
        fudge_factor (float, optional): Factor to scale distances by. Sometimes improves meshing stability.
            Defaults to 0.9.

    Returns:
        function: sdf.d3.sdf3 object defining distance to the surface of the fractal. 
    """

    # Define the underlying SDF
    def distance(p):

        # Claim the lock--the quaternion library is not thread-safe. 
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
            mask =     np.zeros(p.shape[:-1],dtype=bool)    # Mask of points that have crossed the bailout threshold
            new_mask = np.zeros(p.shape[:-1],dtype=bool)    # Mask of points that have crossed the bailout threshold this iteration

            for _ in range(iterations):
                # Update points
                z_1, zp_1 = update(z[~mask])
                zp[~mask] = zp_1 * zp[~mask]
                
                z[~mask] = z_1
                z2[~mask] = mag2(z[~mask])

                # Update mask
                new_mask = (z2 > bailout) & ~mask
                mask = mask | new_mask

            zp2 = mag2(zp) # Calculate magnitude
            zp2 = np.clip(zp2, min=1e-6) # Throw out singular points
            dist = np.sqrt(z2/zp2)*np.log(z2)/(2*power) # Estimate distance 
            dist[~mask] = interior_epsilon # Flatten interior 

            return (dist - offset) * fudge_factor

    return distance

###############################################
# Optimized pipeline (polynomials only, fast) #
###############################################

# For Julia sets with a simple polynomial map (as are generated in the GUI)
# we can use this streamlined SDF implementation which leverages the power
# of numba. Unfortunately we need to redefine our quaternion operations, 
# and can no longer use an arbitrary function for the update rule. 

@njit(inline='always')
def _qmul(w1, x1, y1, z1, w2, x2, y2, z2):
    """
    Qaternion multiplication based on floats. 
    Be careful! Quaternion multiplication is non-commutative. 

    Args:
        w1...z1 (float): Components of quaternion 1
        x2...z2 (float): Components of quaternion 2

    Returns:
        w...z: Components of product.
    """
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return w, x, y, z

@njit(parallel=True, cache=True)
def _polynomial_julia_kernel(points: np.ndarray, coeffs: np.ndarray, slice_w: float, power: int, iterations: int, bailout: float,
                              offset: float, interior_epsilon: float, fudge_factor: float, out: np.ndarray):
    """
    Internal function to calculate distances from a polynomial Julia set.

    Args:
        points (np.ndarray): [N,3] array of points at which to evaluate SDF. 
        out (np.ndarray): [N] array in which to store resultant distances. 
        coeffs (np.ndarray): [M,4] Polynomial coefficients of update rule, lowest-order first.
            Each coefficient is itself a quaternion, specified by its components.  
        slice_w (float): Slice in fourth dimension at which to evaluate SDF.
        power (int): Highest power in polynomial update (used in distance estimate.)
        iterations (int): How many iterations to use to compute the distance estimate. Higher values give 
            more intricate shapes and more accurate distance estimates.
        bailout (float): Value at which to stop iterations for diverging points.
        offset (int): Offset from the fractal surface at which to define the SDF 0-level.
        interior_epsilon (float): Distance to return for points on the interior of the fractal.
            This should be smaller than the meshing grid size to avoid missing chunks.
        fudge_factor (float): Factor to scale distances by. Sometimes improves meshing stability.

    Returns: None (output is stored in `out` vector.)
        
    """
    
    n_terms = coeffs.shape[0]

    # njit gives us the vectorization advantages, so we'll just loop over the points. 
    for i in prange(points.shape[0]):

        # Compute the coordinates of the point in 4D.
        zw = points[i, 0]; zx = points[i, 1]; zy = points[i, 2]; zz = slice_w
        zpw = 1.0; zpx = 0.0; zpy = 0.0; zpz = 0.0  # running derivative, starts at the identity
        escaped = False 
        z2 = 0.0 # Running squared-magnitude of point. 

        for _ in range(iterations):
            # Evaluate the polynomial and its derivative at the current z via
            # an incrementally-tracked running power, instead of recomputing
            # z**i from scratch per term.
            pow_w = 1.0; pow_x = 0.0; pow_y = 0.0; pow_z = 0.0
            prev_w = 0.0; prev_x = 0.0; prev_y = 0.0; prev_z = 0.0
            z1w = 0.0; z1x = 0.0; z1y = 0.0; z1z = 0.0
            zp1w = 0.0; zp1x = 0.0; zp1y = 0.0; zp1z = 0.0

            for t in range(n_terms):
                cw = coeffs[t, 0]; cx = coeffs[t, 1]; cy = coeffs[t, 2]; cz = coeffs[t, 3]
                mw, mx, my, mz = _qmul(pow_w, pow_x, pow_y, pow_z, cw, cx, cy, cz)
                z1w += mw; z1x += mx; z1y += my; z1z += mz
                if t >= 1:
                    dw, dx, dy, dz = _qmul(prev_w, prev_x, prev_y, prev_z, cw, cx, cy, cz)
                    zp1w += t*dw; zp1x += t*dx; zp1y += t*dy; zp1z += t*dz
                prev_w, prev_x, prev_y, prev_z = pow_w, pow_x, pow_y, pow_z
                if t < n_terms - 1:
                    pow_w, pow_x, pow_y, pow_z = _qmul(pow_w, pow_x, pow_y, pow_z, zw, zx, zy, zz)

            zpw, zpx, zpy, zpz = _qmul(zp1w, zp1x, zp1y, zp1z, zpw, zpx, zpy, zpz)
            zw, zx, zy, zz = z1w, z1x, z1y, z1z
            z2 = zw*zw + zx*zx + zy*zy + zz*zz
            if z2 > bailout:
                escaped = True
                break

        if escaped:
            zp2 = zpw*zpw + zpx*zpx + zpy*zpy + zpz*zpz
            if zp2 < 1e-6:
                zp2 = 1e-6
            dist = np.sqrt(z2/zp2) * np.log(z2) / (2*power)
        else:
            dist = interior_epsilon

        out[i] = (dist - offset) * fudge_factor


@d3.sdf3
def polynomial_julia_sdf(coefficients: np.ndarray,
                        slice: float = 0, power: int = 2, iterations: int = 50, bailout: float = 10000**2,
                        offset=0, interior_epsilon = 1e-3, fudge_factor = 0.9):
    """
    Implements approximate distance function for a Julia set with a polynomial update function

    Args:
        coefficients (np.ndarray): M,4] Polynomial coefficients of update rule, lowest-order first.
            Each coefficient is itself a quaternion, specified by its components.
        slice (float, optional): Slice in fourth dimension at which to evaluate SDF. Defaults to 0.
        power (int, optional): Highest power in polynomial update (used in distance estimate.). Defaults to 2.
        iterations (int, optional): How many iterations to use to compute the distance estimate. Higher values give 
            more intricate shapes and more accurate distance estimates. Defaults to 50.
        bailout (float, optional): Value at which to stop iterations for diverging points.. Defaults to 10000**2.
        offset (int, optional): Offset from the fractal surface at which to define the SDF 0-level. Defaults to 0.
        interior_epsilon (_type_, optional): Distance to return for points on the interior of the fractal.
            This should be smaller than the meshing grid size to avoid missing chunks. Defaults to 1e-3.
        fudge_factor (float, optional): Factor to scale distances by. Sometimes improves meshing stability. Defaults to 0.9.

    Returns:
        function: sdf.d3.sdf3 object defining distance to the surface of the fractal. 
    """

    # memory optimization suggested by Claude.
    coeffs = np.ascontiguousarray(coefficients, dtype=np.float64)

    def distance(p):

        # Tidy up memory representation of evaluation points
        p = np.ascontiguousarray(p, dtype=np.float64)

        # Allocate the output array
        out = np.empty(p.shape[0], dtype=np.float64)

        # Compute distances using njit iteration. 
        _polynomial_julia_kernel(p, coeffs, float(slice), float(power), int(iterations),
                                  float(bailout), float(offset), float(interior_epsilon),
                                  float(fudge_factor), out)
        
        return out

    return distance