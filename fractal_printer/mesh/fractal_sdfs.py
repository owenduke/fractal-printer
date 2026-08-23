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
def _qmul(ax, ay, az, aw, bx, by, bz, bw):
    """
    Qaternion multiplication based on floats. 
    Be careful! Quaternion multiplication is non-commutative. 

    Args:
        ax...aw (float): Components of quaternion 1
        bx...bw (float): Components of quaternion 2

    Returns:
        w...z: Components of product.
    """

    x = ax*bx - ay*by - az*bz - aw*bw
    y = ax*by + ay*bx + az*bw - aw*bz
    z = ax*bz + az*bx + aw*by - ay*bw
    w = ax*bw + aw*bx + ay*bz - az*by

    return x, y, z, w

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
        z_x = points[i, 0]; z_y = points[i, 1]; z_z = points[i, 2]; z_w = slice_w
        zp_x = 1.0; zp_y = 0.0; zp_z = 0.0; zp_w = 0.0  # running derivative, starts at the identity
        escaped = False 
        z2 = 0.0 # Running squared-magnitude of point. 

        for _ in range(iterations):
            # Evaluate the polynomial and its derivative at the current z via
            # an incrementally-tracked running power, instead of recomputing
            # z**i from scratch per term.
            pow_x = 1.0; pow_y = 0.0; pow_z = 0.0; pow_w = 0.0
            prev_x = 0.0; prev_y = 0.0; prev_z = 0.0; prev_w = 0.0
            z1_x = 0.0; z1_y = 0.0; z1_z = 0.0; z1_w = 0.0
            zp1_x = 0.0; zp1_y = 0.0; zp1_z = 0.0; zp1_w = 0.0

            for t in range(n_terms):
                c_x = coeffs[t, 0]; c_y = coeffs[t, 1]; c_z = coeffs[t, 2]; c_w = coeffs[t, 3]
                m_x, m_y, m_z, m_w = _qmul(pow_x, pow_y, pow_z, pow_w, c_x, c_y, c_z, c_w)
                z1_x += m_x; z1_y += m_y; z1_z += m_z; z1_w += m_w
                if t >= 1:
                    d_x, d_y, d_z, d_w = _qmul(prev_x, prev_y, prev_z, prev_w, c_x, c_y, c_z, c_w)
                    zp1_x += t*d_x; zp1_y += t*d_y; zp1_z += t*d_z; zp1_w += t*d_w
                prev_x, prev_y, prev_z, prev_w = pow_x, pow_y, pow_z, pow_w
                if t < n_terms - 1:
                    pow_x, pow_y, pow_z, pow_w = _qmul(pow_x, pow_y, pow_z, pow_w, z_x, z_y, z_z, z_w)

            zp_x, zp_y, zp_z, zp_w = _qmul(zp1_x, zp1_y, zp1_z, zp1_w, zp_x, zp_y, zp_z, zp_w)
            z_x, z_y, z_z, z_w = z1_x, z1_y, z1_z, z1_w
            z2 = z_x*z_x + z_y*z_y + z_z*z_z + z_w*z_w
            if z2 > bailout:
                escaped = True
                break

        zp2 = zp_x*zp_x + zp_y*zp_y + zp_z*zp_z + zp_w*zp_w
        if zp2 < 1e-6:
            zp2 = 1e-6
        dist = np.sqrt(z2/zp2) * np.log(z2) / (2*power)

        if dist < -interior_epsilon:
            dist = -interior_epsilon


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