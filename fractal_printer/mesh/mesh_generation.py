# Helper functions for mesh generation
import numpy as np
import meshio
from sdf import core as sdf_core
from sdf.core import _mesh
import fast_simplification



def simplify_mesh(input_mesh, reduction_factor=0.9, target_count = None, aggression = 2, lossless = False):
    points, faces = fast_simplification.simplify(
        points = input_mesh.points, 
        triangles = input_mesh.cells[0].data,
        target_reduction = reduction_factor,
        agg = aggression,
        lossless = lossless
    )
    return meshio.Mesh(points, cells={"triangle": faces})

def box_bounds(size = 2.8):
    return ((-size/2,)*3,(size/2,)*3)



def _bisect_edges(sdf, points, origin, step, tol=1e-8, recursion_levels=30, bracket_tol=1e-3, verbose=True):
    """Bisect marching-cubes vertices against the SDF, stopping each point as soon as its
    value converges OR its bracket has shrunk well below one grid cell (QUIJIBO-style)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return points

    origin = np.asarray(origin, dtype=float)
    step = np.broadcast_to(np.asarray(step, dtype=float), (3,))
    rows = np.arange(n)

    # Every marching-cubes edge is axis-aligned, so exactly one coordinate is a
    # fractional grid index (the interpolated one); the other two sit on the grid.
    ratio = (points - origin) / step
    rounded = np.round(ratio)
    frac = ratio - rounded
    axis = np.argmax(np.abs(frac), axis=1)
    max_frac = np.abs(frac[rows, axis])

    lower, upper = rounded.copy(), rounded.copy()
    lower[rows, axis] = np.floor(ratio[rows, axis])
    upper[rows, axis] = lower[rows, axis] + 1
    corner_a = origin + lower * step
    corner_b = origin + upper * step

    val_a = np.asarray(sdf(corner_a)).reshape(-1)
    val_b = np.asarray(sdf(corner_b)).reshape(-1)

    a_is_pos = val_a >= 0
    pos = np.where(a_is_pos[:, None], corner_a, corner_b)
    neg = np.where(a_is_pos[:, None], corner_b, corner_a)
    pos_val = np.where(a_is_pos, val_a, val_b)
    neg_val = np.where(a_is_pos, val_b, val_a)

    # Grid-aligned/same-sign/non-finite edges keep their linear-interpolation
    # vertex (soft fallback vs QUIJIBO's hard abort on a sign mismatch).
    active = (
        (max_frac > 1e-6) & (pos_val >= 0) & (neg_val < 0)
        & np.isfinite(pos_val) & np.isfinite(neg_val)
    )
    n_skipped = n - int(np.count_nonzero(active))
    if verbose and n_skipped:
        print(f'  {n_skipped} of {n} edges skipped (degenerate/non-finite), kept linear interpolation')

    result = points.copy()
    spatial_tol = np.min(step) * bracket_tol
    for _ in range(recursion_levels):

        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        if verbose:
            print(f'\tLevel {_}: {len(idx)} active edges')

        # A bracket already narrower than sub-grid precision needs no further
        # (expensive) SDF call -- this is what caps the cost of cusp points that
        # never satisfy the value tolerance, instead of burning recursion_levels.
        width = np.linalg.norm(pos[idx] - neg[idx], axis=1)
        tight = width < spatial_tol
        result[idx[tight]] = (pos[idx[tight]] + neg[idx[tight]]) / 2
        active[idx[tight]] = False
        idx = idx[~tight]
        if len(idx) == 0:
            continue

        mid = (pos[idx] + neg[idx]) / 2
        mid_val = np.asarray(sdf(mid)).reshape(-1)
        result[idx] = mid

        converged = np.abs(mid_val) < tol
        active[idx[converged]] = False

        go_neg = (mid_val < 0) & ~converged
        neg[idx[go_neg]] = mid[go_neg]
        go_pos = ~go_neg & ~converged
        pos[idx[go_pos]] = mid[go_pos]

    return result


def generate_bisecting(sdf, samples=2**24, bounds=box_bounds(), recursion_levels=30,
                        tol=1e-8, bracket_tol=1e-3, batch_workers=1, verbose=True):
    if bounds is None:
        bounds = sdf_core._estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds
    volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
    step = (volume / samples) ** (1 / 3)

    # step is passed explicitly (rather than samples) so we know the exact grid
    # spacing used, needed to locate each vertex's edge for bisection below.
    # batch_workers defaults to 1 (no thread-based batch dispatch): a numba
    # sdf3 (e.g. polynomial_julia_sdf) already parallelizes internally via
    # prange across all cores, so an outer ThreadPool here would just contend
    # with that instead of adding anything. Raise it only for a plain,
    # single-threaded sdf where overlapping batches might still help.
    raw_points = np.asarray(
        sdf_core.generate(sdf, step=step, bounds=bounds, workers=batch_workers, verbose=verbose),
        dtype=float,
    )

    # A closed mesh's raw triangle soup repeats each vertex ~6x (once per
    # incident triangle); dedup before bisecting so each distinct edge crossing
    # is refined once instead of redundantly for every triangle sharing it.
    unique_points, inverse = np.unique(raw_points, axis=0, return_inverse=True)

    if verbose:
        print(f'Bisecting {len(unique_points)} unique edge crossings '
              f'(deduped from {len(raw_points)}, up to {recursion_levels} levels, tol={tol})...')
        
    if recursion_levels > 0:
        refined = _bisect_edges(sdf, unique_points, origin=(x0, y0, z0), step=step, tol=tol,
                                recursion_levels=recursion_levels, bracket_tol=bracket_tol, verbose=verbose)
        return refined[inverse]
    else:
        return unique_points[inverse]


def generate_mesh(sdf, samples=2**24, bounds=box_bounds(), recursion_levels=30,
                             tol=1e-8, bracket_tol=1e-3, batch_workers=1,
                             simplify=None, save_path=None, verbose=True):

    # Generate the point list, refining edge crossings against the true SDF
    # instead of trusting marching cubes' linear interpolation between samples.
    points = generate_bisecting(sdf, samples=samples, bounds=bounds, recursion_levels=recursion_levels,
                                 tol=tol, bracket_tol=bracket_tol, batch_workers=batch_workers, verbose=verbose)

    # Convert to meshio Mesh
    print("Converting mesh...")
    mesh = _mesh(points)

    # Optionally simplify
    if simplify is not None:
        print(f"Simplifying mesh by {simplify}x ...")
        mesh = simplify_mesh(mesh, reduction_factor=simplify)

    # Optionally save
    if save_path is not None:
        print(f"Saving mesh to {save_path}...")
        mesh.write(save_path)

    return mesh


