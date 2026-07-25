# Multiprocess (real OS processes, not threads) mesh generation for polynomial
# quaternion Julia sets.
#
# sdf.sdf.core.generate() parallelizes with a ThreadPool, which shares one
# interpreter/GIL across workers. numpy-quaternion's C extension isn't
# thread-safe under that (see fractal_sdfs._QUATERNION_LOCK), so that path
# serializes all quaternion work through a lock -- correct, but it gives up
# essentially all of the parallelism, since the quaternion iteration is ~97%
# of a batch's cost (marching cubes is noise by comparison).
#
# Real processes have no shared state to corrupt, so there's nothing to lock:
# each worker gets its own interpreter and its own copy of the quaternion C
# extension. The catch is that multiprocessing.Pool ships work to workers by
# pickling it, and polynomial_julia_sdf's distance() is a nested closure --
# closures aren't picklable. PolynomialJuliaSDF below is a plain, picklable
# stand-in: it carries only plain data (coefficients, power, iterations, ...)
# and lazily rebuilds the real SDF (via fractal_sdfs.polynomial_julia_sdf,
# the single source of truth for the math) the first time it's called in
# each worker process.
#
# This only covers the polynomial-coefficient case (fractal_sdfs.polynomial_julia_sdf
# / general_julia_sdf with polynomial_update) -- not the general "arbitrary
# update function" path, since an arbitrary user-supplied update function is
# just as likely to be an unpicklable closure.

import itertools
import multiprocessing
import time
from functools import partial

import numpy as np

from sdf.sdf import core
from fractal_printer.mesh import fractal_sdfs as fs
from fractal_printer.mesh.mesh_generation import box_bounds, simplify_mesh


class PolynomialJuliaSDF:
    """Picklable stand-in for fractal_sdfs.polynomial_julia_sdf(...).

    Holds only plain data so multiprocessing can ship it to worker
    processes; each worker lazily builds the real (closure-based) SDF the
    first time it's called, and that build is never itself pickled.
    """

    def __init__(self, coefficients, slice=0, power=2, iterations=50,
                 bailout=10000**2, offset=0, fudge_factor=0.9):
        self.coefficients = coefficients
        self.slice = slice
        self.power = power
        self.iterations = iterations
        self.bailout = bailout
        self.offset = offset
        self.fudge_factor = fudge_factor
        self._sdf = None

    def _ensure_built(self):
        if self._sdf is None:
            self._sdf = fs.polynomial_julia_sdf(
                self.coefficients,
                slice=self.slice,
                power=self.power,
                iterations=self.iterations,
                bailout=self.bailout,
                offset=self.offset,
                fudge_factor=self.fudge_factor,
            )
        return self._sdf

    def __call__(self, p):
        return self._ensure_built()(p)

    def __getstate__(self):
        # Never pickle a built SDF (it's an unpicklable closure) -- each
        # process rebuilds its own on first use instead.
        state = self.__dict__.copy()
        state["_sdf"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def polynomial_julia_sdf_fast(coefficients, **kwargs):
    return PolynomialJuliaSDF(coefficients, **kwargs)


def generate_fast(sdf_spec, step=None, bounds=None, samples=core.SAMPLES,
                   workers=None, batch_size=core.BATCH_SIZE, verbose=True, sparse=True):
    """Same batching/marching-cubes pipeline as sdf.sdf.core.generate, but
    parallelized with real processes instead of a ThreadPool."""

    start = time.time()

    if workers is None:
        workers = multiprocessing.cpu_count()

    if bounds is None:
        bounds = core._estimate_bounds(sdf_spec)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None and samples is not None:
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    if verbose:
        print('min %g, %g, %g' % (x0, y0, z0))
        print('max %g, %g, %g' % (x1, y1, z1))
        print('step %g, %g, %g' % (dx, dy, dz))

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    s = batch_size
    Xs = [X[i:i+s+1] for i in range(0, len(X), s)]
    Ys = [Y[i:i+s+1] for i in range(0, len(Y), s)]
    Zs = [Z[i:i+s+1] for i in range(0, len(Z), s)]

    batches = list(itertools.product(Xs, Ys, Zs))
    num_batches = len(batches)
    num_samples = sum(len(xs) * len(ys) * len(zs) for xs, ys, zs in batches)

    if verbose:
        print('%d samples in %d batches with %d workers' %
              (num_samples, num_batches, workers))

    points = []
    skipped = empty = nonempty = 0
    bar = core.progress.Bar(num_batches, enabled=verbose)
    with multiprocessing.Pool(workers) as pool:
        f = partial(core._worker, sdf_spec, step=(dx, dy, dz), sparse=sparse)
        for result in pool.imap(f, batches):
            bar.increment(1)
            if result is None:
                skipped += 1
            elif len(result) == 0:
                empty += 1
            else:
                nonempty += 1
                points.extend(result)
    bar.done()

    if verbose:
        print('%d skipped, %d empty, %d nonempty' % (skipped, empty, nonempty))
        triangles = len(points) // 3
        seconds = time.time() - start
        print('%d triangles in %g seconds' % (triangles, seconds))

    return points


def generate_mesh_fast(sdf_spec, samples=2**24, bounds=box_bounds(), simplify=None,
                        save_path=None, verbose=True, workers=None, batch_size=core.BATCH_SIZE):
    """Drop-in replacement for mesh_generation.generate_mesh, backed by
    generate_fast's process pool instead of sdf.generate's thread pool."""

    points = generate_fast(sdf_spec, bounds=bounds, samples=samples,
                            workers=workers, batch_size=batch_size, verbose=verbose)

    mesh = core._mesh(points)

    if simplify is not None:
        if verbose:
            print(f"Simplifying mesh by {simplify}x ...")
        mesh = simplify_mesh(mesh, reduction_factor=simplify)

    if save_path is not None:
        if verbose:
            print(f"Saving mesh to {save_path}...")
        mesh.write(save_path)

    return mesh
