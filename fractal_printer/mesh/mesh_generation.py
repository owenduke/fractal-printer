# Helper functions for mesh generation
import meshio
from sdf.sdf.core import _mesh
import fast_simplification


def simplify_mesh(input_mesh, reduction_factor=0.9):
    points, faces = fast_simplification.simplify(
        points = input_mesh.points, 
        triangles = input_mesh.cells[0].data,
        target_reduction = reduction_factor
    )
    return meshio.Mesh(points, cells={"triangle": faces})

def box_bounds(size = 2.8):
    return ((-size/2,)*3,(size/2,)*3)


def generate_mesh(sdf, samples = 2**24, bounds = box_bounds(), simplify = None, save_path = None, verbose=True):
   
    # Generate the point list with SDF's machinery
    points = sdf.generate(bounds = bounds, samples = samples)

    # Convert to meshio Mesh
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


