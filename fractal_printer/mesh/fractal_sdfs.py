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
from numba import njit, prange
from sdf import d3

# numpy-quaternion's C extension isn't thread-safe: concurrent calls into it
# (as happens when sdf.generate's ThreadPool workers each evaluate a quaternion
# SDF) corrupt its internal dtype refcounting and crash the interpreter. Every
# quaternion operation below is serialized through this lock so the threaded
# generation path stays crash-free without giving up parallelism for the
# non-quaternion work (marching cubes, array prep) that runs alongside it.
# Only general_julia_sdf's arbitrary-update path uses numpy-quaternion --
# polynomial_julia_sdf below is a numba kernel and needs no such lock.
_QUATERNION_LOCK = threading.Lock()


def mag2(z):
    return quaternion.as_float_array(z * z.conj())[...,0]


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
def general_julia_sdf(update, slice=0, power = 2, iterations = 50, bailout = 10000**2, offset=0, interior_epsilon = 1e-3, fudge_factor = 0.9):

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
                
                z[~mask] = z_1
                z2[~mask] = mag2(z[~mask])

                # Update mask
                new_mask = (z2 > bailout) & ~mask
                mask = mask | new_mask

            zp2 = mag2(zp)
            zp2 = np.clip(zp2, min=1e-6)
            dist = np.sqrt(z2/zp2)*np.log(z2)/(2*power)
            dist[~mask] = interior_epsilon
            return (dist - offset) * fudge_factor

    return distance

@njit(inline='always')
def _qmul(w1, x1, y1, z1, w2, x2, y2, z2):
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return w, x, y, z


@njit(parallel=True, cache=True)
def _polynomial_julia_kernel(points, coeffs, slice_w, power, iterations, bailout,
                              offset, interior_epsilon, fudge_factor, out):
    n_terms = coeffs.shape[0]
    for i in prange(points.shape[0]):
        # z starts as the point lifted into quaternion space (matches
        # general_julia_sdf's convention: point coords first, slice last).
        zw = points[i, 0]; zx = points[i, 1]; zy = points[i, 2]; zz = slice_w
        zpw = 1.0; zpx = 0.0; zpy = 0.0; zpz = 0.0  # running derivative, starts at the identity
        escaped = False
        z2 = 0.0

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
def polynomial_julia_sdf(coefficients, slice=0, power=2, iterations=50, bailout=10000**2,
                          offset=0, interior_epsilon=1e-3, fudge_factor=0.9):
    coeffs = np.ascontiguousarray(coefficients, dtype=np.float64)

    def distance(p):
        p = np.ascontiguousarray(p, dtype=np.float64)
        out = np.empty(p.shape[0], dtype=np.float64)
        _polynomial_julia_kernel(p, coeffs, float(slice), float(power), int(iterations),
                                  float(bailout), float(offset), float(interior_epsilon),
                                  float(fudge_factor), out)
        return out

    return distance