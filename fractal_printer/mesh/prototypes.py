"""
Shelved experiments in post-hoc mesh refinement, plus the measurement harness built to
evaluate them. Nothing here is imported by the working pipeline.

WHY THIS IS PARKED
------------------
The goal was to recover the sharp ridges that marching cubes chamfers and that
fast_simplification then flattens further, without paying 8x memory per octave of grid
resolution. The approach: pick edges whose chord deviates from the isosurface, split them,
and project the new vertex onto the surface. Every measurable statistic improved -- surface
error, triangle shape, watertightness all held -- and the result still looked worse than
the unrefined mesh. That gap between "the metrics improved" and "it looks worse" was never
closed, so the code is parked here rather than deleted.

The most likely explanation, and the thing to test first if this is picked up again: the
refinement can only redistribute the *existing* connectivity. Where the base grid missed a
ridge entirely, its topology is already wrong, and pulling vertices onto the true surface
inside a wrong topology produces a crumpled surface rather than a sharp one -- locally
closer to the isosurface (which is what the error metric rewards) and globally worse
(which is what the eye sees). The verbose output quantifies how common that is: on the
storm mesh ~3.5% of edges sit further from the surface than a tenth of their own length,
meaning the surface is nowhere near where the mesh thinks it is.

WHAT WAS TRIED, AND WHAT THE MEASUREMENTS SAID
----------------------------------------------
Test case throughout: outputs/storm_refinement.ply, the storm settings from
notebooks/general_julia.ipynb -- 57,528 v / 114,144 t at simplify=0.99. Note that
simplification is what gives that mesh its 906 boundary edges, 85 non-manifold edges and
44 duplicate faces; the input is not clean, and any refinement pass has to survive that.

1. Selection by |sdf(midpoint)|. FAILED, and failed in the specific way of preferring the
   *flattest* regions. A Julia distance estimator is not a distance function: its median
   ratio to the true distance here is 0.17, and the factor varies across the surface,
   being tighter where the escape-time structure is simple. Rank correlation of
   |sdf(midpoint)| with L^2 * curvature was +0.006 -- no signal at all.

2. Selection by dihedral angle across an edge. WEAK: rank correlation with true deviation
   +0.15. The cylinder counter-example is the intuition: axis-parallel edges carry all the
   dihedral but already lie on the surface, while the circumferential edges that actually
   need splitting have none. Scoring whole *faces* and splitting all three edges rescues it
   at roughly 3x the triangles, which is what `criterion='dihedral'` still does.

3. Selection by measured displacement ('sagitta', the default below). WORKED as a
   selector. Walking the midpoint to the surface uses only the *sign* of the sdf, which is
   exact, so it sidesteps (1) entirely. Selection quality, comparing the true deviation of
   the top-12k edges against the rest: |sdf| 3.8x, dihedral 1.7x, measured displacement
   6.7x.

4. Unconstrained projection. BROKE THE GEOMETRY -- this was the visible failure the user
   reported: intersecting faces, vertices flung into space, slivers. Cause was the march
   cap at 1.0x edge length. The median legitimate deviation is 5% of an edge length (p90
   17%), but 1.4% of selected edges travelled more than half an edge and landed on a
   neighbouring sheet of the fractal. Four rounds produced 27,938 crossing face pairs
   (1.82% of surface area) and a p99 error of x1.019 -- worse than not refining.

5. Tight march cap. THE DOMINANT FIX. A real chord deviation is about L^2 * curvature / 8,
   so under L/8 on any patch the grid resolved; anything near L/2 is a different sheet.
   Crossing pairs after four rounds, by cap: 1.0 -> 27938, 0.30 -> 347, 0.20 -> 46,
   0.10 -> 4, 0.05 -> 0. Accuracy *improves* as the cap tightens, because the wild
   landings were themselves the error.

6. Fold guard (_relax_folds). WORKS, MOSTLY REDUNDANT. Guarantees no output triangle turns
   more than acos(min_dot) from the face it was cut from. At cap 0.1 it fires on ~70
   vertices per round, so the cap is doing the real work; kept because it converts "folds
   are unlikely" into "folds are impossible".

7. Longest-edge (Rivara) closure. WORKS. The high-aspect triangles were the split
   *pattern*, not the placement: repeatedly bisecting one edge of a triangle drives the
   median min-angle from 32.3 to 25.6 over four rounds. Longest-edge closure holds it at
   31.8, at the cost of ~3x the triangles of no closure.

8. Tangential Laplacian relaxation of the new vertices. NEGATIVE RESULT, not kept. Median
   error x1.33 and p99 x1.35 *worse*, though min-angle improved 32.3 -> 37.7. The vertices
   are already bisected onto the surface, so smoothing only slides them along the ridge
   toward the flat -- ordinary Laplacian feature shrinkage.

9. Splitting edges whose midpoint finds no surface inside the cap. NOT WORTH IT. Their new
   vertex has to stay at the midpoint, which changes no geometry, so it tripled the
   triangle count for a marginal error gain. Off by default, still available as
   `split_unreachable=True`, and the count is always reported.

Final state on storm, 4 rounds at tolerance=2e-4: 303,580 t (+166%), min-angle median
31.8 (input 32.3), 4 crossing pairs, boundary 906 and non-manifold 85 both unchanged from
the input, 0 degenerate. Clean generated shapes (sphere, torus, sphere-box, with and
without simplify) come out 0 boundary / 0 non-manifold / 0 degenerate / 0 crossings.

A MEASUREMENT TRAP WORTH REMEMBERING
------------------------------------
Two metrics in this file exist because their naive versions gave confidently wrong
answers, and both wrong answers were believed for a while:

  * `surface_error` weights samples by triangle *area*. Sampling uniformly per triangle
    instead makes refinement look like it is making things worse, because refinement adds
    many small triangles in exactly the hardest regions and thereby oversamples them.

  * `count_self_intersections` tests six segment-triangle crossings and deliberately does
    not count coplanar contact. The obvious alternative -- Moller's interval overlap with
    tolerances scaled by the plane distances -- reports adjacent sub-triangles as crossing
    once subdivision has stopped them sharing a vertex index. That inflated the count
    enough to reverse the ranking of the thing being measured, and it also missed genuine
    crossings. `validate_intersection_tester()` is the regression check: exact midpoint
    subdivision cannot change a surface, so it must not change the verdict.

IDEAS NOT TRIED
---------------
  * Slab-streamed generation, so the base grid can be fine enough that the topology is
    right in the first place. The exact integer edge keys in mesh_generation mean slab
    seams weld exactly, so this looks tractable and addresses the root cause rather than
    the symptom.
  * Two-level / Transvoxel adaptive batch resolution -- fine grid only near flagged cells.
  * Dual contouring, which can place a vertex on a crease rather than chamfering it.
    Previously ruled out on watertightness grounds; Manifold Dual Contouring is the
    variant that would need evaluating.
"""

import numpy as np
import meshio
from sdf.d3 import sdf3

from .mesh_generation import CHUNK_SIZE


#######################################################
# Feature-targeted refinement                         #
#######################################################

# Marching cubes cannot put a vertex on a sharp crease: its vertices live on grid edges,
# so a crease comes out as a chamfer strip roughly one cell wide. Raising the grid
# resolution fixes that at 8x the memory per octave, for a feature set that is only
# ~1-D. Refining just the chamfer instead costs O(crease length) and converges on the
# crease geometrically -- each round halves the remaining error.
#
# The whole pass is edge-centric. Faces are *scored*, but it is edges that get marked,
# and a face is then retriangulated purely from which of its three edges carry a new
# vertex. Two faces sharing an edge therefore cannot disagree about whether it was
# split, so T-junctions are impossible by construction and the mesh stays watertight
# without any red-green closure pass.

# Retriangulation for each 3-bit split mask (bit s set == slot s's edge was split).
# Symbols 0,1,2 are the face's own corners; 3,4,5 are the new midpoints of slots 0,1,2.
# Slot 0 spans corners (0,1), slot 1 spans (1,2), slot 2 spans (2,0). Every entry keeps
# the input winding.
_TRI_TABLE = {
    0: ((0, 1, 2),),
    1: ((0, 3, 2), (3, 1, 2)),
    2: ((0, 1, 4), (0, 4, 2)),
    3: ((3, 1, 4), (0, 3, 4), (0, 4, 2)),
    4: ((0, 1, 5), (5, 1, 2)),
    5: ((0, 3, 5), (3, 1, 2), (3, 2, 5)),
    6: ((4, 2, 5), (0, 1, 4), (0, 4, 5)),
    7: ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)),
}


def _edge_topology(points: np.ndarray, faces: np.ndarray) -> tuple:
    """
    Build the edge table a refinement round needs, in one argsort.

    The alternative -- scanning the face list for each face's neighbours -- is O(F^2).
    Measured on a 745k-face mesh that is ~280 s even as a parallel numba kernel, against
    0.41 s here.

    Args:
        points (np.ndarray): [V,3] vertex positions.
        faces (np.ndarray): [F,3] triangle indices.

    Returns:
        tuple: (slot_edge [F,3] edge id per face slot, ends [E,2] endpoint indices,
            dihedral [E] 1 - cos(angle) across each edge (0 where undefined),
            march [E,3] unit sum of incident face normals,
            regular [E] bool, True where the edge is safe to subdivide, n_edges)
    """
    V, F = len(points), len(faces)

    # Rows 0..F-1 are slot 0, F..2F-1 slot 1, 2F..3F-1 slot 2.
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    key = e[:, 0].astype(np.int64) * V + e[:, 1]
    uniq, flat_edge, counts = np.unique(key, return_inverse=True, return_counts=True)
    flat_edge = flat_edge.reshape(-1)
    n_edges = len(uniq)
    slot_edge = flat_edge.reshape(3, F).T
    ends = np.stack([uniq // V, uniq % V], axis=1)

    # Unit face normals. Normalising once here makes every dihedral below a bare dot
    # product -- cheaper than a law-of-cosines form and, unlike sin, monotonic in the
    # angle all the way to 180 degrees.
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    normal = np.cross(v1 - v0, v2 - v0)
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-30)[:, None]

    # Dihedral, defined only where exactly two faces meet. Boundary and non-manifold
    # edges score 0 so they never *trigger* a split, but they still split correctly if a
    # neighbouring face is flagged, since the split mask is per-edge.
    dihedral = np.zeros(n_edges, dtype=np.float64)
    order = np.argsort(flat_edge, kind='stable')
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pair = counts == 2
    if pair.any():
        s0, s1 = order[starts[pair]], order[starts[pair] + 1]
        dihedral[pair] = 1.0 - np.einsum('ij,ij->i', normal[s0 % F], normal[s1 % F]).clip(-1, 1)

    # March direction: the summed normal of every face on the edge. Across a chamfer the
    # two faces straddle the crease, so their sum points along the bisector -- straight
    # at the feature. In a flat region they agree and it is just the surface normal.
    face_of_slot = np.tile(np.arange(F), 3)
    march = np.empty((n_edges, 3), dtype=np.float64)
    for c in range(3):
        march[:, c] = np.bincount(flat_edge, weights=normal[face_of_slot, c], minlength=n_edges)
    march /= np.maximum(np.linalg.norm(march, axis=1), 1e-30)[:, None]

    # An edge is safe to subdivide only if exactly two faces meet along it AND neither
    # is one of a duplicated pair. A pair of identical faces contributes two incidences
    # by itself, so an edge interior to the pair reads as manifold, passes a naive
    # count == 2 test, and then has both copies emit the same interior edges -- four
    # faces on each. fast_simplification produces such pairs (89 faces in 114k on the
    # storm mesh), so this is not hypothetical.
    _, inverse, repeats = np.unique(np.sort(faces, axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    duplicated = repeats[inverse.reshape(-1)] > 1
    tainted = np.bincount(flat_edge, weights=duplicated[face_of_slot].astype(np.float64),
                          minlength=n_edges) > 0
    regular = pair & ~tainted

    return slot_edge, ends, dihedral, march, regular, n_edges


def _project_to_surface(sdf: sdf3, start: np.ndarray, direction: np.ndarray, cap: np.ndarray,
                        march_steps: int = 8, bisect_steps: int = 24, tol: float = 1e-10,
                        chunk: int = CHUNK_SIZE) -> tuple:
    """
    Walk each point along its direction until the SDF changes sign, then bisect.

    A split edge's midpoint is the average of two points already on the surface, so near
    a convex crease it sits inside the solid and near a concave one outside. The march
    direction is taken from the sign of the SDF there rather than assumed.

    The walk is capped, and the cap has to be tight. On a fractal there is another sheet
    of surface a short distance away in almost every direction, so a generous cap does
    not merely waste evaluations -- it finds the wrong sheet and welds the new vertex
    onto it. A chord of length L across a patch of curvature radius R deviates by about
    L^2/8R, so a legitimate deviation is under L/8 wherever the patch is resolved at all;
    anything approaching L/2 is a different sheet, not a sharper crease.

    The caller is told which of the three outcomes happened rather than just whether the
    point moved, because "no sign change inside the cap" is informative: it means the
    surface is further away than the cap, so the chord is *badly* wrong even though we
    decline to guess where it should go.

    Args:
        sdf (sdf3): sdf.d3.sdf3 object defining the mesh.
        start (np.ndarray): [M,3] starting points.
        direction (np.ndarray): [M,3] unit directions, oriented outward.
        cap (np.ndarray): [M] maximum distance to travel.
        march_steps (int, optional): Probes along the capped ray. Defaults to 8.
        bisect_steps (int, optional): Bisections once bracketed. Defaults to 24.
        tol (float, optional): |sdf| below which a point is done. Defaults to 1e-10.
        chunk (int, optional): Points per SDF call. Defaults to CHUNK_SIZE.

    Returns:
        tuple: (points [M,3] projected positions, status [M] int8 -- 0 already on the
            surface (or the sdf was not finite there), 1 bracketed and moved, 2 no sign
            change inside the cap, left where it started)
    """
    def evaluate(pts):
        out = np.empty(len(pts), dtype=np.float64)
        for lo in range(0, len(pts), chunk):
            hi = min(lo + chunk, len(pts))
            out[lo:hi] = np.asarray(sdf(pts[lo:hi])).reshape(-1)
        return out

    m = len(start)
    result = start.copy()
    status = np.zeros(m, dtype=np.int8)
    if m == 0:
        return result, status

    d0 = evaluate(start)
    sign0 = np.sign(d0)

    # Points already on the surface, or where the SDF is not finite, stay put.
    live = np.isfinite(d0) & (np.abs(d0) > tol) & (sign0 != 0)
    ray = -sign0[:, None] * direction          # toward the surface

    lo_t = np.zeros(m, dtype=np.float64)
    hi_t = np.zeros(m, dtype=np.float64)
    hit = np.zeros(m, dtype=bool)

    for k in range(1, march_steps + 1):
        active = np.flatnonzero(live & ~hit)
        if len(active) == 0:
            break
        t = cap[active] * (k / march_steps)
        d = evaluate(start[active] + ray[active] * t[:, None])
        flipped = np.isfinite(d) & (np.sign(d) != sign0[active])
        hi_t[active[flipped]] = t[flipped]
        hit[active[flipped]] = True
        lo_t[active[~flipped]] = t[~flipped]

    idx = np.flatnonzero(hit)
    if len(idx):
        a, b = lo_t[idx].copy(), hi_t[idx].copy()
        alive = np.ones(len(idx), dtype=bool)
        for _ in range(bisect_steps):
            sel = np.flatnonzero(alive)
            if len(sel) == 0:
                break
            mid_t = (a[sel] + b[sel]) / 2
            pts = start[idx[sel]] + ray[idx[sel]] * mid_t[:, None]
            d = evaluate(pts)
            done = np.abs(d) < tol
            same = np.sign(d) == sign0[idx[sel]]
            a[sel[same]] = mid_t[same]
            b[sel[~same]] = mid_t[~same]
            alive[sel[done]] = False
        result[idx] = start[idx] + ray[idx] * ((a + b) / 2)[:, None]

    status[hit] = 1
    status[live & ~hit] = 2
    return result, status


def _retriangulate(faces: np.ndarray, slot_edge: np.ndarray, split: np.ndarray,
                   new_index: np.ndarray) -> tuple:
    """
    Rebuild every face from its 3-bit mask of split edges.

    Args:
        faces (np.ndarray): [F,3] input triangles.
        slot_edge (np.ndarray): [F,3] edge id per face slot, from _edge_topology.
        split (np.ndarray): [E] bool, which edges carry a new vertex.
        new_index (np.ndarray): [E] vertex index of each edge's new vertex, -1 if none.

    Returns:
        tuple: (faces [F',3] output triangles, parent [F'] index of the face each came from)
    """
    # corner[:, 0:3] are the face's own vertices, corner[:, 3:6] the slot midpoints,
    # which is exactly the symbol numbering _TRI_TABLE indexes into.
    corner = np.concatenate([faces, new_index[slot_edge]], axis=1)
    mask = split[slot_edge] @ np.array([1, 2, 4], dtype=np.int64)

    pieces, parent = [], []
    for code, tris in _TRI_TABLE.items():
        rows_of = np.flatnonzero(mask == code)
        if len(rows_of) == 0:
            continue
        rows = corner[rows_of]
        for tri in tris:
            pieces.append(rows[:, tri])
            parent.append(rows_of)
    return np.concatenate(pieces), np.concatenate(parent)


def _relax_folds(points: np.ndarray, faces: np.ndarray, midpoint: np.ndarray,
                 offset: np.ndarray, new_faces: np.ndarray, parent: np.ndarray,
                 min_dot: float = 0.5, steps: int = 10) -> tuple:
    """
    Back each new vertex off along its own displacement until nothing folds.

    Projecting a midpoint onto the isosurface is the whole point of the pass, but the
    displacement is unconstrained: a new vertex can travel far enough to swing one of the
    triangles it sits on past its neighbours, and on a mesh with ridges packed a few edge
    lengths apart that reads as intersecting faces, spikes and slivers. Adjacent split
    edges make it worse, since their midpoints move independently and can cross.

    The fix is to treat the input face as the reference orientation. Plain midpoint
    subdivision is exactly coplanar with its parent, so a displacement scale of 0 is
    always safe; scale 1 may not be. Halving the scale of every new vertex on an offending
    triangle and re-testing therefore converges, and the last iteration snaps whatever is
    still bad to 0. The output cannot contain a triangle turned more than
    acos(min_dot) from the face it came from -- no inversions, and no zero-area spikes.

    Args:
        points (np.ndarray): [V,3] input vertex positions.
        faces (np.ndarray): [F,3] input triangles, used only for reference normals.
        midpoint (np.ndarray): [N,3] undisplaced midpoints of the split edges.
        offset (np.ndarray): [N,3] proposed displacement of each new vertex.
        new_faces (np.ndarray): [F',3] output triangles, from _retriangulate.
        parent (np.ndarray): [F'] input face each output triangle came from.
        min_dot (float, optional): Smallest allowed cos(angle) between an output triangle's
            normal and its parent's. Defaults to 0.5 (60 degrees).
        steps (int, optional): Halvings before a stubborn vertex is snapped back to its
            midpoint. Defaults to 10.

    Returns:
        tuple: (positions [N,3] of the new vertices, scale [N] the fraction of the
            proposed displacement each one kept)
    """
    normal = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                      points[faces[:, 2]] - points[faces[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-30)[:, None]
    reference = normal[parent]

    base = len(points)
    scale = np.ones(len(midpoint), dtype=np.float64)

    def offenders():
        """Output triangles turned more than acos(min_dot) from their parent."""
        pos = np.concatenate([points, midpoint + scale[:, None] * offset])
        a = pos[new_faces[:, 0]]
        n = np.cross(pos[new_faces[:, 1]] - a, pos[new_faces[:, 2]] - a)
        # Compare against min_dot * |n| rather than normalising: same test, no divide, and
        # a collapsed triangle (|n| == 0) fails it automatically.
        return np.einsum('ij,ij->i', n, reference) < min_dot * np.linalg.norm(n, axis=1)

    def new_vertices_of(bad):
        touched = new_faces[bad].reshape(-1)
        return np.unique(touched[touched >= base] - base)

    # Every face has to be re-tested each pass, not just the ones that were bad: pulling a
    # vertex back changes both faces it sits on, and the one that was fine is not
    # guaranteed to stay fine.
    for _ in range(steps):
        bad = offenders()
        if not bad.any():
            return midpoint + scale[:, None] * offset, scale
        scale[new_vertices_of(bad)] *= 0.5

    # Halving has run out of patience, so snap the remaining offenders all the way back.
    # This terminates: a face whose new vertices are all at scale 0 is an exact midpoint
    # subdivision of its parent, so every pass either zeroes at least one more vertex or
    # finds nothing left to zero. The second case is not dead code -- with min_dot at 1.0
    # the test is failed by rounding alone, and without the progress check the loop would
    # spin once per new vertex.
    for _ in range(len(midpoint) + 1):
        bad = offenders()
        if not bad.any():
            break
        stubborn = new_vertices_of(bad)
        stubborn = stubborn[scale[stubborn] > 0]
        if len(stubborn) == 0:
            break
        scale[stubborn] = 0.0

    return midpoint + scale[:, None] * offset, scale


def refine_mesh(input_mesh: meshio.Mesh, sdf: sdf3, tolerance: float = None,
                criterion: str = 'sagitta', threshold: float = 0.5, rounds: int = 6,
                closure: str = 'longest', split_unreachable: bool = False,
                max_march: float = 0.1, min_dot: float = 0.5, relax_steps: int = 10,
                march_steps: int = 8, bisect_steps: int = 24,
                tol: float = 1e-10, chunk: int = CHUNK_SIZE, verbose: bool = True) -> meshio.Mesh:
    """
    Sharpen creases by subdividing only the faces that sit on one, then pushing the new
    vertices onto the isosurface.

    Two criteria are available.

    'sagitta' (default) walks each edge's midpoint onto the surface and splits the edge
    if it had to travel further than `tolerance`. That displacement is the chord's real
    deviation, in model units, so the tolerance can come straight off a print resolution.
    It is curvature-adaptive for free (on a smooth patch the deviation is L^2 * curvature
    / 8) and anisotropic where it counts: an edge running along a ridge already lies on
    the surface and is left alone, while one crossing the ridge is split.

    Note that this *measures* the displacement rather than reading |sdf(midpoint)|. A
    Julia distance estimator is not a distance function -- on the storm mesh its median
    ratio to the true distance is 0.17, and that factor varies across the surface, being
    tighter where the escape-time structure is simple. Thresholding the raw value
    therefore preferentially selects smooth regions, the opposite of what is wanted. The
    walk uses only the *sign* of the sdf, which is exact.

    'dihedral' scores each face by the Pythagorean sum of `1 - cos(angle)` over its three
    edges and splits all three edges of a flagged face. This suits genuinely sharp
    creases, which reach the mesh as a chamfer strip whose *boundary* edges carry the
    large dihedrals. Note that dihedral *across* an edge is a poor per-edge predictor of
    deviation (rank correlation +0.15 on the storm mesh): on a cylinder the axis-parallel
    edges carry all the dihedral yet already lie on the surface, while the circumferential
    edges that need splitting have none. Scoring faces and splitting all three edges is
    what rescues it, at roughly three times the triangles.

    One vertex is created per marked edge, and each face is then rebuilt from its 3-bit
    mask of split edges (0/1/2/3 edges split -> 1/2/3/4 triangles). Because the mask is
    read off the shared edge list, neighbours always agree, so no closure pass is needed
    and the mesh cannot develop T-junctions.

    Placement is the part that has to be defended. Subdividing is always safe -- midpoints
    are coplanar with the face they came from -- but moving the new vertex onto the surface
    is not, and an unconstrained projection on a fractal produces exactly the failure the
    pass is supposed to prevent: vertices that march across a gap onto a neighbouring
    sheet, faces that fold through each other, and slivers. Two guards run:

      * the march is capped at `max_march` times the edge length, which is a bound on the
        chord deviation a well-sampled patch can have (see _project_to_surface); and
      * every new vertex is then pulled back along its own displacement until no output
        triangle is turned more than acos(`min_dot`) from the face it came from
        (_relax_folds), which is a hard no-inversion guarantee.

    An edge whose midpoint finds no surface inside the cap is still *split* -- the chord is
    known to be badly wrong -- but its new vertex stays at the plain midpoint, and the
    shorter edges of the next round usually resolve it.

    Args:
        input_mesh (meshio.Mesh): Starting mesh.
        sdf (sdf3): The sdf.d3.sdf3 object the mesh was generated from.
        tolerance (float, optional): For criterion='sagitta', the largest chord deviation
            to accept: an edge is split while its midpoint sits further than this from the
            surface. A distance in model units, so it can be set from a physical print
            resolution. Required for the sagitta criterion. Defaults to None.
        criterion (str, optional): 'sagitta' or 'dihedral'. Defaults to 'sagitta'.
        threshold (float, optional): For criterion='dihedral', face score above which to
            refine. The score runs 0 (flat) to 2*sqrt(3); 0.5 catches a face with one
            60-degree neighbour or two 45-degree ones. Defaults to 0.5.
        rounds (int, optional): Maximum passes. The sagitta criterion halts on its own
            once every edge is inside `tolerance`, so this is only a backstop for it.
            Defaults to 6.
        closure (str, optional): How to stop repeated bisection from producing slivers.
            'longest' also marks the longest edge of every face that has any marked edge
            (Rivara), 'red' promotes any face with two or three marked edges to a full
            1 -> 4, 'none' bisects exactly what was selected. Defaults to 'longest'.
        split_unreachable (bool, optional): Whether to split edges whose midpoint found no
            surface inside the march cap. Their new vertex has to stay at the midpoint, so
            it buys nothing this round and only pays off if a later round's shorter edges
            reach the surface. Defaults to False; the count is reported either way.
        max_march (float, optional): How far a midpoint may travel, as a multiple of its
            edge's length. Raising this does not buy sharper creases -- a real chord
            deviation is around L^2 * curvature / 8, well under L/8 on any patch the grid
            resolved -- it just lets a midpoint reach a neighbouring sheet of the fractal.
            Defaults to 0.1. On the storm mesh, four rounds at 1.0 leave 27938 crossing
            face pairs; 0.2 leaves 46 and 0.1 leaves 4, and the surface error is *better*
            at 0.1 than at 1.0 either way. Tighten `tolerance` for accuracy, not this.
        min_dot (float, optional): Smallest cos(angle) allowed between an output triangle
            and the face it was cut from. 1.0 disables projection entirely, 0.0 permits
            anything short of an inversion. Defaults to 0.5, i.e. 60 degrees.
        relax_steps (int, optional): Halvings a fold-guarded vertex gets before it is
            snapped back to its midpoint. Defaults to 10.
        march_steps (int, optional): Probes along the capped ray. Defaults to 8.
        bisect_steps (int, optional): Bisections once bracketed. Defaults to 24.
        tol (float, optional): |sdf| below which a vertex is considered on the surface.
            Defaults to 1e-10.
        chunk (int, optional): Points per SDF call. Defaults to CHUNK_SIZE.
        verbose (bool, optional): Whether to output progress messages. Defaults to True.

    Returns:
        meshio.Mesh: Refined mesh, in the input's dtypes.
    """
    point_dtype = input_mesh.points.dtype
    face_dtype = input_mesh.cells[0].data.dtype
    points = input_mesh.points.astype(np.float64, copy=True)
    faces = input_mesh.cells[0].data.astype(np.int64, copy=True)

    if criterion not in ('sagitta', 'dihedral'):
        raise ValueError("criterion must be 'sagitta' or 'dihedral'")
    if criterion == 'sagitta' and tolerance is None:
        raise ValueError("criterion='sagitta' needs a tolerance (max chord deviation)")

    for r in range(rounds):
        slot_edge, ends, dihedral, march, regular, n_edges = _edge_topology(points, faces)

        precomputed = None
        if criterion == 'sagitta':
            # Measure every candidate edge's deviation by walking its midpoint to the
            # surface, then keep the walk: the edges we select have already been placed,
            # so selection costs only the walks that turn out to be unnecessary.
            cand = np.flatnonzero(regular)
            mid = (points[ends[cand, 0]] + points[ends[cand, 1]]) / 2
            span = np.linalg.norm(points[ends[cand, 1]] - points[ends[cand, 0]], axis=1)
            cap = np.maximum(span, 1e-12) * max_march
            landed, status = _project_to_surface(
                sdf, mid, march[cand], cap,
                march_steps=march_steps, bisect_steps=bisect_steps, tol=tol, chunk=chunk)
            deviation = np.zeros(n_edges, dtype=np.float64)
            deviation[cand] = np.linalg.norm(landed - mid, axis=1)
            # An edge that found no surface inside its cap is over tolerance by
            # definition -- its midpoint is at least `cap` from the surface -- but its new
            # vertex would have to stay at the midpoint, which changes no geometry at all,
            # so splitting it only pays off in a later round. Off by default. The
            # cap > tolerance term is what makes it terminate when it is on: once an edge
            # is short enough that even a full march could not clear the tolerance,
            # splitting it again cannot help.
            unreachable = (status == 2) & (cap > tolerance)
            keep = (status == 1) & (deviation[cand] > tolerance)
            if split_unreachable:
                keep |= unreachable
            split = np.zeros(n_edges, dtype=bool)
            split[cand[keep]] = True
            precomputed = (cand, landed, status)
        else:
            score = np.sqrt((dihedral[slot_edge] ** 2).sum(axis=1))
            flagged = score > threshold
            split = np.zeros(n_edges, dtype=bool)
            split[slot_edge[flagged].reshape(-1)] = True

        # Leave topologically broken neighbourhoods alone. Splitting a boundary or
        # non-manifold edge just yields two of the same defect, so a mesh that arrives
        # broken -- fast_simplification readily produces holes, non-manifold edges and
        # duplicate faces -- would have its defect count double every round. A face
        # whose third edge is held back simply falls to the two-split case, which the
        # table already handles.
        held = int((split & ~regular).sum())
        split &= regular

        # Closure. Bisecting one edge of a triangle ('green') is the cheapest way to keep
        # the mesh conforming, but it is also how aspect ratio rots: the two halves are
        # thinner than the parent, and after a few rounds the same triangles have been
        # bisected repeatedly. Two standard remedies, both of which only ever *add* to the
        # split set, so the mask stays consistent between neighbours.
        if closure != 'none':
            length = np.linalg.norm(points[ends[:, 1]] - points[ends[:, 0]], axis=1)
            for _ in range(32):
                marked = split[slot_edge]
                if closure == 'red':
                    # Any face with two or three split edges is promoted to a full 1 -> 4.
                    extra = slot_edge[marked.sum(axis=1) >= 2].reshape(-1)
                else:
                    # Rivara: a face is only ever bisected from its longest edge, which is
                    # what bounds the smallest angle over unlimited refinement.
                    rows = slot_edge[marked.any(axis=1)]
                    extra = rows[np.arange(len(rows)), np.argmax(length[rows], axis=1)]
                more = np.zeros(n_edges, dtype=bool)
                more[extra] = True
                more &= regular
                if not (more & ~split).any():
                    break
                split |= more

        n_split = int(split.sum())

        if verbose:
            if criterion == 'sagitta':
                what = (f'{n_split} of {n_edges} edges split '
                        f'(deviation median {np.median(deviation[deviation > 0]):.3e}, '
                        f'max {deviation.max():.3e}; {int(unreachable.sum())} edges further '
                        f'off than the march cap, '
                        f'{"included" if split_unreachable else "skipped"})')
            else:
                what = (f'{int(flagged.sum())} of {len(faces)} faces flagged '
                        f'(score > {threshold}), {n_split} of {n_edges} edges split')
            print(f'  Round {r}: ' + what
                  + (f', {held} held back at boundary/non-manifold edges' if held else ''))
        if n_split == 0:
            break

        sel = np.flatnonzero(split)
        a, b = ends[sel, 0], ends[sel, 1]
        midpoint = (points[a] + points[b]) / 2

        if precomputed is not None:
            # The selection pass already walked these midpoints; reuse the landings.
            cand, landed, status = precomputed
            where = np.searchsorted(cand, sel)
            moved, hit = landed[where], status[where] == 1
        else:
            length = np.linalg.norm(points[b] - points[a], axis=1)
            moved, status = _project_to_surface(
                sdf, midpoint, march[sel], length * max_march,
                march_steps=march_steps, bisect_steps=bisect_steps, tol=tol, chunk=chunk)
            hit = status == 1

        # Only a bracketed walk earns a displacement; everything else stays at the
        # midpoint, which is always a safe place to put a new vertex.
        offset = np.where(hit[:, None], moved - midpoint, 0.0)

        # New vertices are appended, so existing indices stay valid.
        new_index = np.full(n_edges, -1, dtype=np.int64)
        new_index[sel] = len(points) + np.arange(n_split)
        new_faces, parent = _retriangulate(faces, slot_edge, split, new_index)

        placed, scale = _relax_folds(points, faces, midpoint, offset, new_faces, parent,
                                     min_dot=min_dot, steps=relax_steps)
        if verbose:
            shift = np.linalg.norm(placed - midpoint, axis=1)
            # Only count vertices that actually had a displacement to lose.
            pulled = (scale < 1) & hit
            print(f'    projected {int(hit.sum())}/{n_split} new vertices '
                  f'(median shift {np.median(shift[hit]) if hit.any() else 0:.3e}, '
                  f'{n_split - int(hit.sum())} left at the midpoint), '
                  f'{int(pulled.sum())} pulled back by the fold guard '
                  f'({int((pulled & (scale == 0)).sum())} of them all the way)')

        points = np.concatenate([points, placed])
        faces = new_faces

        if verbose:
            print(f'    -> {len(points)} vertices, {len(faces)} triangles')

    return meshio.Mesh(points.astype(point_dtype, copy=False),
                       cells={"triangle": faces.astype(face_dtype, copy=False)})


#######################################################
# Convenience wrapper                                 #
#######################################################

def generate_and_refine(sdf: sdf3, refine: float = None, refine_criterion: str = 'sagitta',
                        refine_rounds: int = 6, **generate_kwargs) -> meshio.Mesh:
    """
    generate_mesh followed by refine_mesh, which is how the pass was wired in when it
    lived in mesh_generation.

    Refinement runs last, after any simplification: it is cheapest on the smallest mesh,
    and simplification would otherwise collapse the very triangles it just added.

    Args:
        sdf (sdf3): sdf.d3.sdf3 object defining the mesh.
        refine (float, optional): Chord tolerance for criterion='sagitta', or the face
            score threshold for 'dihedral'. None skips refinement. Defaults to None.
        refine_criterion (str, optional): 'sagitta' or 'dihedral'. Defaults to 'sagitta'.
        refine_rounds (int, optional): Maximum refinement passes. Defaults to 6.
        **generate_kwargs: Passed straight through to generate_mesh.

    Returns:
        meshio.Mesh: Refined mesh.
    """
    from .mesh_generation import generate_mesh

    save_path = generate_kwargs.pop('save_path', None)
    verbose = generate_kwargs.get('verbose', True)
    mesh = generate_mesh(sdf, **generate_kwargs)

    if refine is not None:
        if verbose:
            print(f"Refining ({refine_criterion}, {refine}, up to {refine_rounds} round(s)) ...")
        keyword = 'tolerance' if refine_criterion == 'sagitta' else 'threshold'
        mesh = refine_mesh(mesh, sdf, criterion=refine_criterion, rounds=refine_rounds,
                           chunk=generate_kwargs.get('chunk_size', CHUNK_SIZE),
                           verbose=verbose, **{keyword: refine})

    if save_path is not None:
        if verbose:
            print(f"Saving to {save_path} ...")
        mesh.write(save_path)
    return mesh


#######################################################
# Measurement harness                                 #
#######################################################

# Barycentric sample points, deliberately away from the corners: the vertices are bisected
# onto the isosurface by construction, so sampling near them measures nothing. The chamfer
# error lives in the triangle interiors.
_BARY = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6],
                  [.5, .25, .25], [.25, .5, .25], [.25, .25, .5]])

# Relative barycentric / parametric margin in the intersection test. Below this a pair is
# in contact rather than penetrating.
_MARGIN = 1e-9


def surface_error(sdf: sdf3, points: np.ndarray, faces: np.ndarray,
                  quantiles: tuple = (0.5, 0.99), chunk: int = CHUNK_SIZE) -> tuple:
    """
    |sdf| sampled over triangle interiors, weighted by triangle area.

    The area weighting is the whole point. Sampling uniformly *per triangle* biases the
    result against refinement: a refined mesh has many small triangles concentrated in
    exactly the regions that were hardest, so per-triangle sampling oversamples them and
    reports the error getting worse when it is getting better.

    Args:
        sdf (sdf3): The sdf.d3.sdf3 object the mesh approximates.
        points (np.ndarray): [V,3] vertex positions.
        faces (np.ndarray): [F,3] triangle indices.
        quantiles (tuple, optional): Area-weighted quantiles to return. Defaults to
            (0.5, 0.99).
        chunk (int, optional): Points per SDF call. Defaults to CHUNK_SIZE.

    Returns:
        tuple: |sdf| at each requested quantile, in model units.
    """
    tri = np.asarray(points, np.float64)[np.asarray(faces, np.int64)]
    pts = np.einsum('bk,nkc->nbc', _BARY, tri).reshape(-1, 3)

    d = np.empty(len(pts), dtype=np.float64)
    for lo in range(0, len(pts), chunk):
        hi = min(lo + chunk, len(pts))
        d[lo:hi] = np.asarray(sdf(pts[lo:hi])).reshape(-1)

    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    weight = np.repeat(area, len(_BARY))

    magnitude = np.abs(d)
    order = np.argsort(magnitude)
    cumulative = np.cumsum(weight[order])
    cumulative /= cumulative[-1]
    return tuple(magnitude[order][np.searchsorted(cumulative, q)] for q in quantiles)


def min_angles(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Smallest angle of each triangle, in degrees.

    The smallest angle is opposite the shortest side, so sorting the side lengths lets one
    law-of-cosines evaluation cover the whole mesh.

    Args:
        points (np.ndarray): [V,3] vertex positions.
        faces (np.ndarray): [F,3] triangle indices.

    Returns:
        np.ndarray: [F] smallest angle per triangle, in degrees.
    """
    p = np.asarray(points, np.float64)
    f = np.asarray(faces, np.int64)
    sides = np.sort(np.stack([np.linalg.norm(p[f[:, 1]] - p[f[:, 2]], axis=1),
                              np.linalg.norm(p[f[:, 2]] - p[f[:, 0]], axis=1),
                              np.linalg.norm(p[f[:, 0]] - p[f[:, 1]], axis=1)], axis=1), axis=1)
    short, mid, longest = sides[:, 0], sides[:, 1], sides[:, 2]
    cosine = (mid ** 2 + longest ** 2 - short ** 2) / np.maximum(2 * mid * longest, 1e-300)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def audit_topology(points: np.ndarray, faces: np.ndarray) -> dict:
    """
    Count the defects that stop a mesh printing.

    Args:
        points (np.ndarray): [V,3] vertex positions.
        faces (np.ndarray): [F,3] triangle indices.

    Returns:
        dict: boundary (edges used by one face), non_manifold (used by more than two),
            degenerate (a repeated vertex index), zero_area, duplicate_faces.
    """
    p = np.asarray(points, np.float64)
    f = np.asarray(faces, np.int64)
    V = len(p)

    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e[:, 0].astype(np.int64) * V + e[:, 1], return_counts=True)

    normal = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    _, repeats = np.unique(np.sort(f, axis=1), axis=0, return_counts=True)

    return dict(
        boundary=int((counts == 1).sum()),
        non_manifold=int((counts > 2).sum()),
        degenerate=int(((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])).sum()),
        zero_area=int((np.linalg.norm(normal, axis=1) < 1e-24).sum()),
        duplicate_faces=int((repeats > 1).sum()),
    )


def _segment_hits_triangle(origin: np.ndarray, direction: np.ndarray, a: np.ndarray,
                           b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Moller-Trumbore, strict: does origin -> origin+direction cross the interior of abc?

    Args:
        origin (np.ndarray): [n,3] segment starts.
        direction (np.ndarray): [n,3] segment vectors.
        a (np.ndarray): [n,3] first triangle corners.
        b (np.ndarray): [n,3] second triangle corners.
        c (np.ndarray): [n,3] third triangle corners.

    Returns:
        np.ndarray: [n] bool.
    """
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, p)

    # Parallel or degenerate is not a crossing. Scale the cutoff by the magnitudes actually
    # involved so it means the same thing at any model size.
    scale = (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
             * np.linalg.norm(direction, axis=1))
    live = np.abs(det) > 1e-12 * np.maximum(scale, 1e-300)
    inverse = np.where(live, 1.0 / np.where(live, det, 1.0), 0.0)

    to_a = origin - a
    u = np.einsum('ij,ij->i', to_a, p) * inverse
    q = np.cross(to_a, e1)
    v = np.einsum('ij,ij->i', direction, q) * inverse
    t = np.einsum('ij,ij->i', e2, q) * inverse

    m = _MARGIN
    return live & (u > m) & (v > m) & (u + v < 1 - m) & (t > m) & (t < 1 - m)


def _triangles_cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Do these triangle pairs properly intersect? [n,3,3] x [n,3,3] -> [n] bool.

    Tested as six segment-triangle crossings. Coplanar overlap is deliberately *not*
    counted: it is contact rather than penetration, and it is exactly what edge-adjacent
    sub-triangles look like once subdivision has stopped them sharing a vertex index.
    Counting it inflates the total on any subdivided mesh badly enough to reverse the
    ranking of whatever is being compared.

    Args:
        first (np.ndarray): [n,3,3] first triangles of each pair.
        second (np.ndarray): [n,3,3] second triangles of each pair.

    Returns:
        np.ndarray: [n] bool.
    """
    hit = np.zeros(len(first), dtype=bool)
    if len(first) == 0:
        return hit
    for source, target in ((first, second), (second, first)):
        a, b, c = target[:, 0], target[:, 1], target[:, 2]
        for i in range(3):
            j = (i + 1) % 3
            hit |= _segment_hits_triangle(source[:, i], source[:, j] - source[:, i], a, b, c)
    return hit


def _grid_pairs(lo: np.ndarray, hi: np.ndarray, cell: float) -> np.ndarray:
    """
    Broad phase: candidate triangle pairs sharing a cell of a uniform AABB grid hash.

    A KD-tree on centroids is the obvious alternative and is much worse here, because the
    query radius has to cover the largest triangle in the mesh and fractal meshes have a
    wide spread of triangle sizes.

    Args:
        lo (np.ndarray): [F,3] per-triangle bounding-box minima.
        hi (np.ndarray): [F,3] per-triangle bounding-box maxima.
        cell (float): Grid cell size.

    Returns:
        np.ndarray: [n,2] sorted, unique candidate index pairs.
    """
    grid_lo = np.floor(lo / cell).astype(np.int64)
    span = np.floor(hi / cell).astype(np.int64) - grid_lo + 1
    counts = span.prod(1)

    triangle = np.repeat(np.arange(len(lo)), counts)
    offset = np.arange(len(triangle)) - np.repeat(
        np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    ny, nz = span[triangle, 1], span[triangle, 2]
    cz = offset % nz
    rest = offset // nz
    cy = rest % ny
    cx = rest // ny

    key = (((grid_lo[triangle, 0] + cx) * 73856093)
           ^ ((grid_lo[triangle, 1] + cy) * 19349663)
           ^ ((grid_lo[triangle, 2] + cz) * 83492791))
    order = np.argsort(key, kind='stable')
    key, triangle = key[order], triangle[order]

    starts = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    sizes = np.concatenate([starts[1:], [len(key)]]) - starts

    blocks = []
    for size in np.unique(sizes):
        if size < 2:
            continue
        bucket = triangle[starts[sizes == size][:, None] + np.arange(size)[None, :]]
        iu, ju = np.triu_indices(size, 1)
        blocks.append(np.stack([bucket[:, iu].ravel(), bucket[:, ju].ravel()], axis=1))
    if not blocks:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.sort(np.concatenate(blocks), axis=1), axis=0)


def count_self_intersections(points: np.ndarray, faces: np.ndarray,
                             chunk: int = 200000) -> tuple:
    """
    Find triangles that pass through triangles they share no vertex with.

    Args:
        points (np.ndarray): [V,3] vertex positions.
        faces (np.ndarray): [F,3] triangle indices.
        chunk (int, optional): Candidate pairs per narrow-phase batch. Defaults to 200000.

    Returns:
        tuple: (crossing pairs, flagged triangles, fraction of surface area on a flagged
            triangle). The area fraction is the figure to compare across resolutions: one
            crossing between two large triangles becomes many crossing pairs of small ones
            after subdivision, so the pair count is not scale-invariant.
    """
    p = np.asarray(points, np.float64)
    f = np.asarray(faces, np.int64)
    tri = p[f]
    lo, hi = tri.min(1), tri.max(1)

    candidates = _grid_pairs(lo, hi, max(np.percentile(hi - lo, 90), 1e-12))
    left, right = candidates[:, 0], candidates[:, 1]
    candidates = candidates[((lo[left] <= hi[right]) & (lo[right] <= hi[left])).all(1)]
    fl, fr = f[candidates[:, 0]], f[candidates[:, 1]]
    candidates = candidates[~(fl[:, :, None] == fr[:, None, :]).any(2).any(1)]

    flagged = np.zeros(len(f), dtype=bool)
    pairs = 0
    for start in range(0, len(candidates), chunk):
        batch = candidates[start:start + chunk]
        hit = _triangles_cross(p[f[batch[:, 0]]], p[f[batch[:, 1]]])
        pairs += int(hit.sum())
        flagged[batch[hit, 0]] = True
        flagged[batch[hit, 1]] = True

    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return pairs, int(flagged.sum()), float(area[flagged].sum() / area.sum())


def audit_mesh(mesh: meshio.Mesh, sdf: sdf3 = None, label: str = '') -> dict:
    """
    One-line summary of everything worth knowing about a candidate mesh.

    Args:
        mesh (meshio.Mesh): Mesh to audit.
        sdf (sdf3, optional): If given, also measure the area-weighted surface error.
            Defaults to None.
        label (str, optional): Prefix for the printed line. Defaults to ''.

    Returns:
        dict: The measured quantities.
    """
    points = mesh.points.astype(np.float64)
    faces = mesh.cells[0].data.astype(np.int64)

    angles = min_angles(points, faces)
    result = audit_topology(points, faces)
    pairs, flagged, area = count_self_intersections(points, faces)
    result.update(vertices=len(points), triangles=len(faces),
                  min_angle_median=float(np.median(angles)),
                  min_angle_p1=float(np.percentile(angles, 1)),
                  crossing_pairs=pairs, crossing_triangles=flagged, crossing_area=area)
    if sdf is not None:
        median, p99 = surface_error(sdf, points, faces)
        result.update(error_median=median, error_p99=p99)

    print("  %-24s %8d t | minang med %5.2f p1 %5.2f | cross %5d (%.4f%%) | "
          "bnd %4d nm %4d dg %d%s"
          % (label, result['triangles'], result['min_angle_median'], result['min_angle_p1'],
             pairs, 100 * area, result['boundary'], result['non_manifold'],
             result['degenerate'],
             '' if sdf is None else " | err med %.3e p99 %.3e"
             % (result['error_median'], result['error_p99'])))
    return result


def uniform_subdivide(points: np.ndarray, faces: np.ndarray, times: int = 1) -> tuple:
    """
    Exact 1 -> 4 midpoint subdivision, used as a control.

    Midpoints are coplanar with the face they came from, so this cannot change the surface
    and therefore cannot change any true geometric property of it. That makes it the
    reference every metric here has to be invariant under.

    Args:
        points (np.ndarray): [V,3] vertex positions.
        faces (np.ndarray): [F,3] triangle indices.
        times (int, optional): Subdivision rounds. Defaults to 1.

    Returns:
        tuple: (points, faces) of the subdivided mesh.
    """
    points = np.asarray(points, np.float64)
    faces = np.asarray(faces, np.int64)
    for _ in range(times):
        slot_edge, ends, _, _, _, n_edges = _edge_topology(points, faces)
        faces, _ = _retriangulate(faces, slot_edge, np.ones(n_edges, dtype=bool),
                                  len(points) + np.arange(n_edges))
        points = np.concatenate([points, (points[ends[:, 0]] + points[ends[:, 1]]) / 2])
    return points, faces


def validate_intersection_tester(verbose: bool = True) -> bool:
    """
    Regression test for count_self_intersections. An earlier version passed none of these,
    and its numbers were believed long enough to misdirect a whole round of tuning.

    Checks, in order: a genuine crossing is found; coplanar edge contact is not counted;
    and clean generated meshes report zero at every subdivision level, which is the
    invariance that catches tolerance bugs.

    Args:
        verbose (bool, optional): Print each case. Defaults to True.

    Returns:
        bool: True if every case passed.
    """
    from sdf import d3
    from .mesh_generation import generate_mesh, box_bounds

    ok = True

    def check(name, got, want):
        nonlocal ok
        passed = got == want
        ok &= passed
        if verbose:
            print("  %-42s got %-5s want %-5s %s"
                  % (name, got, want, 'ok' if passed else 'FAIL'))

    crossing = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [.3, .3, -.5], [.3, .3, .5], [.9, .9, 0.]], dtype=np.float64)
    check("genuine crossing",
          count_self_intersections(crossing, np.array([[0, 1, 2], [3, 4, 5]]))[0], 1)

    contact = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                        [0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    check("coplanar edge contact is not a crossing",
          count_self_intersections(contact, np.array([[0, 1, 2], [3, 4, 5]]))[0], 0)

    for name, shape in (("sphere", d3.sphere(1.0)),
                        ("torus", d3.torus(1.0, 0.35)),
                        ("sphere-box", d3.sphere(1.0) - d3.box((1.2, 1.2, 1.2)))):
        mesh = generate_mesh(shape, samples=2 ** 18, bounds=box_bounds(2.8),
                             recursion_levels=15, verbose=False)
        p = mesh.points.astype(np.float64)
        f = mesh.cells[0].data.astype(np.int64)
        for level in (0, 1, 2):
            pk, fk = uniform_subdivide(p, f, level) if level else (p, f)
            check("clean %s, subdivided %dx" % (name, level),
                  count_self_intersections(pk, fk)[0], 0)

    if verbose:
        print("  -> %s" % ("all passed" if ok else "FAILURES"))
    return ok
