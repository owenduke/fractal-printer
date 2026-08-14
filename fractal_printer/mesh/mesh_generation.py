# Helper functions for mesh generation
import numpy as np
import meshio
from sdf import core as sdf_core
from sdf.d3 import sdf3
from sdf.core import _mesh
import fast_simplification
from pathlib import Path

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

def _bisect_edges(sdf: sdf3, points: np.ndarray, origin: list, step: float, 
                  tol=1e-8, recursion_levels=30, bracket_tol=1e-3, verbose=True) -> np.ndarray:
    """
    Bisect marching-cubes vertices against the SDF, stopping each point as soon as its
    value converges OR its bracket has shrunk well below one grid cell. 
    Based on (QUIJIBO-style).

    Returns:
        _type_: _description_
    """

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

def generate_mesh(sdf: sdf3, samples: int = 2**24, bounds: tuple=box_bounds(), recursion_levels: int =30,
                             tol: float = 1e-8, bracket_tol: float =1e-3,
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

    # Use the builtin sdf module marching cubes implementation to generate the approximate mesh.
    # Use one worker to avoid conflicting with numba parallelism.  
    raw_points = np.asarray( 
        sdf_core.generate(sdf, step=step, bounds=bounds, workers=1, verbose=verbose),
        dtype=float
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
        points = refined[inverse]
    else:
        points = unique_points[inverse]

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


