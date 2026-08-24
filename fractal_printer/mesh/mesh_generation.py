# Helper functions for mesh generation
import itertools
import time
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import meshio
from skimage import measure
from sdf import core as sdf_core
from sdf import progress as sdf_progress
from sdf.d3 import sdf3
import fast_simplification
from pathlib import Path

# Points per SDF call when refining. Large enough to keep the numba kernel's threads
# busy, small enough that the per-call coordinate temporaries stay off the peak. Sweeps
# at 2**23 and 2**25 samples both put this at the bottom of the memory *and* wall-clock
# curves -- bigger chunks buy nothing and cost cache locality.
CHUNK_SIZE = 2**16


def simplify_mesh(input_mesh: meshio.Mesh, reduction_factor = 0.9, aggression = 2, lossless = False) -> meshio.Mesh:
    """
    Wrapper for the fast_simplification module to use meshio.Mesh objects for input and output.
    See fast_simplification docs for details on the effect of arguments.

    Args:
        input_mesh (meshio.Mesh): Starting mesh.
        reduction_factor (float, optional): How much we want to reduce the triangle count by. Defaults to 0.9.
        aggression (int, optional): Determines how large an angle can be between two faces before merging is blocked.
            Higher values result in more reduction at the cost of loss of detail. Defaults to 2.
        lossless (bool, optional): Whether to do lossless simplification. If true, overrides the other parameters
            and only simplifies the mesh where doing so doesn't change the geometry at all (for instance, if the
            mesh elements all lie in a plane). Defaults to False.

    Returns:
        meshio.Mesh: Simplified mesh.
    """
    points, faces = fast_simplification.simplify(
        points = input_mesh.points,
        triangles = input_mesh.cells[0].data,
        target_reduction = reduction_factor,
        agg = aggression,
        lossless = lossless
    )
    # fast_simplification always hands back float64/int64; carry the input's narrower
    # dtypes through so a float32 mesh doesn't silently double in size here.
    points = points.astype(input_mesh.points.dtype, copy=False)
    faces = faces.astype(input_mesh.cells[0].data.dtype, copy=False)
    return meshio.Mesh(points, cells={"triangle": faces})

def box_bounds(size:float = 2.8) -> tuple:
    """
    Helper function to generate a cube-shaped bounding box centered on the origin.

    Args:
        size (float, optional): Size of the box. Defaults to 2.8.

    Returns:
        tuple: ((-size/2,size/2)*3) coordinates of the box edges.
    """
    return ((-size/2,)*3,(size/2,)*3)


#######################################################
# Edge keys: the compact representation of a vertex   #
#######################################################

# Almost every marching-cubes vertex lies on one axis-aligned grid edge, so it is
# fully described by that edge's identity plus a scalar position along it:
#
#     key = ((i * ny + j) * nz + k) * 4 + slot      (int64)
#     t   = fractional position along `slot`, in [0, 1)
#
# 16 bytes per vertex instead of 24 for float64 xyz, but the real win is that the
# key is an *exact integer*. Deduplication becomes np.unique over a 1-D int64 array
# rather than a lexsort over three float64 columns -- far cheaper, and immune to the
# last-ulp disagreements that can arise when two adjacent batches reconstruct the
# same seam vertex from different batch origins.
#
# `slot` is 0/1/2 for a crossing interpolated along x/y/z. Two cases need more:
#
#   slot == 3   The vertex landed exactly on grid corner (i,j,k) -- every coordinate
#               integral, so no axis is distinguished. It gets its own slot rather
#               than being folded into slot 0, which would collide with the genuine
#               x-edge crossing out of the same corner.
#
#   key  <  0   skimage's Lewiner marching cubes resolves some ambiguous cell
#               configurations by adding a vertex in the cell *interior*, which lies
#               on no edge at all (two or three fractional coordinates). These are
#               rare -- a handful in a 35k-vertex mesh -- and can never be duplicated,
#               since batches partition cells even though they share corner planes.
#               They are stored verbatim in the `extra` side table and referenced by
#               a negative key, -(row + 1).


def _encode_edges(verts: np.ndarray, base_index: tuple, dims: tuple) -> tuple:
    """
    Turn a batch's marching-cubes vertices into (key, t) pairs.

    Args:
        verts (np.ndarray): [n,3] vertex positions in batch-local grid-index coordinates,
            as returned by skimage.measure.marching_cubes.
        base_index (tuple): (i0, j0, k0) global grid index of the batch's first corner.
        dims (tuple): (nx, ny, nz) global grid dimensions.

    Returns:
        tuple: (keys [n] int64, t [n] float64, extra [m,3] float64 grid coordinates of
            the cell-interior vertices, whose keys are set to the -1 sentinel for the
            caller to renumber globally)
    """
    _, ny, nz = dims

    # Promote to global grid indices. An edge crossing is interpolated along exactly one
    # axis, so its other two coordinates are exact integers and survive this untouched.
    v = verts.astype(np.float64, copy=True)
    v += np.asarray(base_index, dtype=np.float64)

    base = np.floor(v)
    frac = v - base
    n_frac = np.count_nonzero(frac, axis=1)
    slot = np.argmax(frac, axis=1)
    t = frac[np.arange(len(v)), slot]

    # No fractional coordinate at all: the vertex is exactly on grid corner (i,j,k).
    slot = np.where(n_frac == 0, 3, slot)

    idx = base.astype(np.int64)
    keys = ((idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]) * 4 + slot

    # More than one fractional coordinate: a Lewiner cell-interior vertex. It has no
    # edge to be keyed by, so hand it back for the side table.
    interior = n_frac > 1
    keys[interior] = -1
    return keys, t, v[interior]


def _edge_points(keys: np.ndarray, t: np.ndarray, origin: np.ndarray,
                 step: np.ndarray, dims: tuple, extra: np.ndarray) -> np.ndarray:
    """
    Decode (key, t) pairs back into world coordinates.

    Args:
        keys (np.ndarray): [n] int64 edge keys.
        t (np.ndarray): [n] positions along each edge. Not restricted to [0,1); passing
            zeros/ones yields the edge's two grid corners. Ignored for corner (slot 3)
            and side-table (negative key) vertices, which have no edge to slide along.
        origin (np.ndarray): [3] world coordinate of grid index (0,0,0).
        step (np.ndarray): [3] grid spacing.
        dims (tuple): (nx, ny, nz) global grid dimensions.
        extra (np.ndarray): [m,3] grid coordinates of the cell-interior vertices.

    Returns:
        np.ndarray: [n,3] float64 world coordinates.
    """
    _, ny, nz = dims
    n = len(keys)

    p = np.empty((n, 3), dtype=np.float64)

    # The side table is nearly always empty, so avoid splitting the arrays when it is.
    on_grid = keys >= 0
    all_on_grid = bool(on_grid.all())
    if all_on_grid:
        kg, tg, q = keys, t, p
    else:
        kg, tg = keys[on_grid], t[on_grid]
        q = np.empty((len(kg), 3), dtype=np.float64)

    # Unpack straight into q's columns and divide `rest` down in place, so only one
    # index temporary is ever live at a time.
    slot = kg & 3                          # keys are non-negative here, so & 3 == % 4
    rest = kg >> 2
    q[:, 2] = rest % nz
    rest //= nz
    q[:, 1] = rest % ny
    rest //= ny
    q[:, 0] = rest
    del rest

    # Offset along the edge. Addressing q flat keeps this to one gather instead of the
    # several a boolean-masked 2-D fancy index would need. Slot 3 sits exactly on the
    # corner, so it is clamped to a valid column and given a zero offset.
    edge = slot < 3
    np.minimum(slot, 2, out=slot)
    flat = np.arange(len(kg), dtype=np.intp)
    flat *= 3
    flat += slot
    q.reshape(-1)[flat] += np.where(edge, tg, 0.0)

    if not all_on_grid:
        p[on_grid] = q
        p[~on_grid] = extra[-keys[~on_grid] - 1]

    p *= step
    p += origin
    return p


def _decode_all(keys: np.ndarray, t: np.ndarray, origin: np.ndarray, step: np.ndarray,
                dims: tuple, extra: np.ndarray, dtype=np.float32,
                chunk: int = CHUNK_SIZE) -> np.ndarray:
    """
    Decode every (key, t) pair straight into an output array of the requested dtype,
    a chunk at a time, so a full-length float64 intermediate is never materialised.

    Returns:
        np.ndarray: [n,3] array of world coordinates in `dtype`.
    """
    n = len(keys)
    out = np.empty((n, 3), dtype=dtype)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        out[lo:hi] = _edge_points(keys[lo:hi], t[lo:hi], origin, step, dims, extra)
    return out


#######################################################
# Surface extraction                                  #
#######################################################

def _surface_worker(sdf: sdf3, job, dims: tuple, sparse: bool):
    """
    Marching cubes over one batch, returning indexed geometry.

    sdf.core's worker ends with `verts[faces].reshape((-1,3))`, which discards skimage's
    already-deduplicated connectivity and inflates the batch ~6x into a triangle soup --
    a soup the caller then has to spend gigabytes putting back together. Here the
    (verts, faces) pair is kept intact and the vertices are handed back as edge keys.

    Args:
        sdf (sdf3): sdf.d3.sdf3 object defining the mesh.
        job (tuple): ((i0, X), (j0, Y), (k0, Z)) index offsets and coordinate arrays.
        dims (tuple): (nx, ny, nz) global grid dimensions.
        sparse (bool): Whether to skip batches the SDF proves are surface-free.

    Returns:
        None if the batch produced no geometry, else
        (keys [n] int64, t [n] float64, faces [f,3] int32, extra [m,3] float64).
    """
    (i0, X), (j0, Y), (k0, Z) = job

    if sparse and sdf_core._skip(sdf, (X, Y, Z)):
        return None

    P = sdf_core._cartesian_product(X, Y, Z)
    volume = sdf(P).reshape((len(X), len(Y), len(Z)))
    del P

    try:
        verts, faces, _, _ = measure.marching_cubes(volume, 0)
    except Exception:
        return None
    if len(faces) == 0:
        return None

    keys, t, extra = _encode_edges(verts, (i0, j0, k0), dims)
    return keys, t, faces.astype(np.int32, copy=False), extra


def _generate_surface(sdf: sdf3, step, bounds: tuple, workers: int = 1, batch_size: int = 32,
                      sparse: bool = True, verbose: bool = True) -> tuple:
    """
    Extract the isosurface as indexed geometry over edge keys.

    Batches are stitched by offsetting each one's face indices by the running vertex
    count, then everything is blitted into single preallocated arrays -- popping each
    piece as it is copied, so the parts are freed during the copy rather than coexisting
    with the result. Vertices are still duplicated across shared batch boundaries at this
    point (~10% of them); the caller's key deduplication removes those.

    Args:
        sdf (sdf3): sdf.d3.sdf3 object defining the mesh.
        step (float | tuple): Grid spacing, scalar or per-axis.
        bounds (tuple): ((x0,y0,z0),(x1,y1,z1)) region to mesh over.
        workers (int, optional): Thread count. Defaults to 1, to avoid contending with
            numba parallelism in the SDF kernel.
        batch_size (int, optional): Grid cells per axis per batch. Defaults to 32.
        sparse (bool, optional): Whether to skip provably surface-free batches. Defaults to True.
        verbose (bool, optional): Whether to output progress messages. Defaults to True.

    Returns:
        tuple: (keys [n] int64, t [n] float64, faces [f,3] int32, extra [m,3] float64,
            origin [3], step [3], dims)
    """
    start = time.time()
    (x0, y0, z0), (x1, y1, z1) = bounds

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
    dims = (len(X), len(Y), len(Z))

    s = batch_size
    Xs = [(i, X[i:i + s + 1]) for i in range(0, len(X), s)]
    Ys = [(i, Y[i:i + s + 1]) for i in range(0, len(Y), s)]
    Zs = [(i, Z[i:i + s + 1]) for i in range(0, len(Z), s)]

    batches = list(itertools.product(Xs, Ys, Zs))
    num_batches = len(batches)
    num_samples = sum(len(xb) * len(yb) * len(zb) for (_, xb), (_, yb), (_, zb) in batches)

    if verbose:
        print('%d samples in %d batches with %d workers' % (num_samples, num_batches, workers))

    key_chunks, t_chunks, face_chunks, extra_chunks = [], [], [], []
    n_verts = n_faces = n_extra = 0
    skipped = nonempty = 0

    bar = sdf_progress.Bar(num_batches, enabled=verbose)
    pool = ThreadPool(workers)
    worker = partial(_surface_worker, sdf, dims=dims, sparse=sparse)
    try:
        for result in pool.imap(worker, batches):
            bar.increment(1)
            if result is None:
                skipped += 1
                continue
            nonempty += 1
            keys, t, faces, extra = result
            faces += np.int32(n_verts)          # stitch onto the running vertex list
            if len(extra):
                # Renumber the batch's -1 sentinels into global side-table references.
                # `extra` was gathered in the same order the sentinels appear.
                keys[keys < 0] = -(np.arange(len(extra), dtype=np.int64) + n_extra + 1)
                extra_chunks.append(extra)
                n_extra += len(extra)
            key_chunks.append(keys)
            t_chunks.append(t)
            face_chunks.append(faces)
            n_verts += len(keys)
            n_faces += len(faces)
    finally:
        pool.close()
        pool.join()
    bar.done()

    all_keys = np.empty(n_verts, dtype=np.int64)
    all_t = np.empty(n_verts, dtype=np.float64)
    all_faces = np.empty((n_faces, 3), dtype=np.int32)

    # imap yields in batch order; reverse so pop() preserves it while freeing as we go.
    key_chunks.reverse(); t_chunks.reverse(); face_chunks.reverse()
    vo = fo = 0
    while key_chunks:
        keys = key_chunks.pop(); t = t_chunks.pop(); faces = face_chunks.pop()
        all_keys[vo:vo + len(keys)] = keys
        all_t[vo:vo + len(t)] = t
        all_faces[fo:fo + len(faces)] = faces
        vo += len(keys); fo += len(faces)

    all_extra = (np.concatenate(extra_chunks) if extra_chunks
                 else np.empty((0, 3), dtype=np.float64))

    if verbose:
        print('%d skipped, %d nonempty' % (skipped, nonempty))
        if n_extra:
            print('%d cell-interior vertices held in the side table' % n_extra)
        print('%d triangles in %g seconds' % (n_faces, time.time() - start))

    origin = np.array([x0, y0, z0], dtype=np.float64)
    step_vec = np.array([dx, dy, dz], dtype=np.float64)
    return all_keys, all_t, all_faces, all_extra, origin, step_vec, dims


#######################################################
# Surface refinement                                  #
#######################################################

def _bisect_edges(sdf: sdf3, keys: np.ndarray, t: np.ndarray, origin: np.ndarray,
                  step: np.ndarray, dims: tuple, extra: np.ndarray, tol: float = 1e-8,
                  recursion_levels: int = 30, bracket_tol: float = 1e-3,
                  chunk: int = CHUNK_SIZE, verbose: bool = True) -> np.ndarray:
    """
    Bisect marching-cubes vertices against the SDF, stopping each point as soon as its
    value converges OR its bracket has shrunk well below one grid cell.
    Based on (QUIJIBO-style).

    Because every vertex lies on a single axis-aligned edge, the bracket is a pair of
    scalars rather than a pair of 3D points: the whole search runs in the 1-D parameter
    t, and world coordinates are only materialised for the chunk being evaluated. That
    replaces eleven live [n,3] float64 arrays with four [n] ones.

    Args:
        sdf (sdf3): sdf.d3.sdf3 object defining the mesh.
        keys (np.ndarray): [n] int64 edge keys.
        t (np.ndarray): [n] linearly-interpolated positions along each edge.
        origin (np.ndarray): [3] world coordinate of grid index (0,0,0).
        step (np.ndarray): [3] grid spacing.
        dims (tuple): (nx, ny, nz) global grid dimensions.
        extra (np.ndarray): [m,3] grid coordinates of the cell-interior vertices.
        tol (float, optional): Bisection tolerance (distance from true surface). Defaults to 1e-8.
        recursion_levels (int, optional): Number of bisections to do. Defaults to 30.
        bracket_tol (float, optional): Bisection tolerance (length of search window),
            as a fraction of the grid step. Defaults to 1e-3.
        chunk (int, optional): Points per SDF call. Defaults to CHUNK_SIZE.
        verbose (bool, optional): Whether to output progress messages. Defaults to True.

    Returns:
        np.ndarray: [n] refined positions along each edge.
    """
    n = len(keys)
    if n == 0:
        return t.astype(np.float64, copy=True)

    def evaluate(sel_keys, sel_t):
        """SDF over a set of (key, t) pairs, chunked to bound the coordinate temporary."""
        out = np.empty(len(sel_keys), dtype=np.float64)
        for lo in range(0, len(sel_keys), chunk):
            hi = min(lo + chunk, len(sel_keys))
            pts = _edge_points(sel_keys[lo:hi], sel_t[lo:hi], origin, step, dims, extra)
            out[lo:hi] = np.asarray(sdf(pts)).reshape(-1)
        return out

    # The two grid corners of every edge are t = 0 and t = 1.
    val_a = evaluate(keys, np.zeros(n))
    val_b = evaluate(keys, np.ones(n))

    a_is_pos = val_a >= 0
    t_pos = np.where(a_is_pos, 0.0, 1.0)
    t_neg = np.where(a_is_pos, 1.0, 0.0)
    pos_val = np.where(a_is_pos, val_a, val_b)
    neg_val = np.where(a_is_pos, val_b, val_a)
    del val_a, val_b, a_is_pos

    # Only genuine edge crossings have a 1-D bracket to search: side-table vertices sit
    # in a cell interior and slot-3 vertices sit on a corner, so both are left where
    # marching cubes put them. Grid-aligned/same-sign/non-finite edges likewise keep
    # their linear-interpolation vertex (soft fallback vs QUIJIBO's hard abort on a sign
    # mismatch).
    on_edge = (keys >= 0) & (keys % 4 < 3)
    active = (
        on_edge & (t > 1e-6) & (t < 1 - 1e-6) & (pos_val >= 0) & (neg_val < 0)
        & np.isfinite(pos_val) & np.isfinite(neg_val)
    )
    del pos_val, neg_val, on_edge

    n_skipped = n - int(np.count_nonzero(active))
    if verbose and n_skipped:
        print(f'  {n_skipped} of {n} edges skipped (degenerate/non-finite), kept linear interpolation')

    result = t.astype(np.float64, copy=True)
    spatial_tol = np.min(step) * bracket_tol

    for level in range(recursion_levels):

        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        if verbose:
            print(f'\tLevel {level}: {len(idx)} active edges')

        # pos and neg differ only along the edge's own axis, so the bracket's spatial
        # width is just the parameter width scaled by that axis' step.
        width = np.abs(t_pos[idx] - t_neg[idx]) * step[keys[idx] % 4]
        tight = width < spatial_tol
        result[idx[tight]] = (t_pos[idx[tight]] + t_neg[idx[tight]]) / 2
        active[idx[tight]] = False
        idx = idx[~tight]
        if len(idx) == 0:
            continue

        mid = (t_pos[idx] + t_neg[idx]) / 2
        mid_val = evaluate(keys[idx], mid)
        result[idx] = mid

        converged = np.abs(mid_val) < tol
        active[idx[converged]] = False

        go_neg = (mid_val < 0) & ~converged
        t_neg[idx[go_neg]] = mid[go_neg]
        go_pos = ~go_neg & ~converged
        t_pos[idx[go_pos]] = mid[go_pos]

    return result


#######################################################
# Top level                                           #
#######################################################

def generate_mesh(sdf: sdf3, samples: int = 2**24, bounds: tuple=box_bounds(), recursion_levels: int =30,
                             tol: float = 1e-8, bracket_tol: float =1e-3, out_dtype=np.float32,
                             chunk_size: int = CHUNK_SIZE,
                             simplify: float = None, save_path: str | Path =None, verbose: bool = True) -> meshio.Mesh:
    """
    Generate a mesh from an sdf object

    Args:
        sdf (sdf3): sdf.dn.sdf3 object defining the mesh.
        samples (int, optional): Number of samples to use for computation. Defaults to 2**24.
        bounds (tuple, optional): Bounds of the region over which to compute the mesh
            (e.g., ((x_min,x_max),(y_min,y_max),...)). Defaults to box_bounds().
        recursion_levels (int, optional): Number of bisections to do. Defaults to 30.
        tol (float, optional): Bisection tolerance (distance from true surface). Defaults to 1e-8.
        bracket_tol (float, optional): Bisection tolerance (length of search window). Defaults to 1e-3.
        out_dtype (np.dtype, optional): Vertex dtype of the returned mesh. Defaults to np.float32,
            which halves vertex memory and is what binary STL stores anyway. Bisection itself
            always runs in float64; pass np.float64 to keep that precision in a .ply.
        chunk_size (int, optional): Points per SDF call during refinement. Lower it to trade
            a little speed for a smaller transient. Defaults to CHUNK_SIZE.
        simplify (float, optional): How much to simplify the resulting mesh. Defaults to None.
        save_path (str | Path, optional): If specified, location at which to save the resultant mesh. Can be
            any format supported by the meshio module (.ply, .stl, ...). Defaults to None.
        verbose (bool, optional): Whether to output progress messages. Defaults to True.

    Returns:
        meshio.Mesh representing the computed shape.
    """

    # Estimate the volume to mesh over. This is spotty for fractal meshes--specify the bounds manually where possible.
    if bounds is None:
        bounds = sdf_core._estimate_bounds(sdf)

    # Calculate the grid step explicitly so we can bisect later
    (x0, y0, z0), (x1, y1, z1) = bounds
    volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
    step = (volume / samples) ** (1 / 3)

    # Marching cubes over the grid, as indexed geometry rather than a triangle soup.
    # One worker, to avoid conflicting with numba parallelism.
    keys, t, faces, extra, origin, step_vec, dims = _generate_surface(
        sdf, step=step, bounds=bounds, workers=1, verbose=verbose)

    # Vertices on a shared batch boundary were emitted by every batch touching them.
    # Deduplicating by exact edge key is a 1-D int64 unique, and it doubles as the
    # remap for the face indices.
    n_raw = len(keys)
    keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    t = t[first]
    del first
    # Narrow before the gather: inverse[faces] at int64 would be twice the size for
    # a result that has to end up int32 anyway.
    inverse = inverse.reshape(-1).astype(np.int32, copy=False)
    faces = inverse[faces]
    del inverse

    if verbose:
        print(f'Bisecting {len(keys)} unique edge crossings '
                f'(deduped from {n_raw}, up to {recursion_levels} levels, tol={tol})...')

    if recursion_levels > 0:
        t = _bisect_edges(sdf, keys, t, origin=origin, step=step_vec, dims=dims, extra=extra,
                          tol=tol, recursion_levels=recursion_levels, bracket_tol=bracket_tol,
                          chunk=chunk_size, verbose=verbose)

    # Convert to meshio Mesh. `faces` is already the connectivity over the deduplicated
    # vertices, so this is just a decode of the compact (key, t) representation.
    print("Converting mesh...")
    mesh = meshio.Mesh(
        _decode_all(keys, t, origin, step_vec, dims, extra, dtype=out_dtype, chunk=chunk_size),
        [("triangle", faces)],
    )
    del keys, t, faces, extra

    # Optionally simplify
    if simplify is not None:
        print(f"Simplifying mesh by {simplify}x ...")
        mesh = simplify_mesh(mesh, reduction_factor=simplify)

    # Optionally save
    if save_path is not None:
        print(f"Saving mesh to {save_path}...")
        mesh.write(save_path)

    return mesh
