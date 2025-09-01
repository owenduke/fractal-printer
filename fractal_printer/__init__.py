import numpy as np
import quaternion
import struct
import fractal_printer.distance_functions as distance_functions
import tqdm

def iterate_voxels(distance_function, levels, size=1, cutoff = 0, kwargs = {}):
    origin = -np.quaternion(0,1,1,1)*size/2
    steps = 3**levels
    resolution = size/steps
    mask = np.zeros((steps,steps,steps),dtype=np.int8)

    for n in range(levels-1,-1,-1):
        stride = 3**n
        span = (stride-1)//2
        points = range(span,steps-span,stride)
        for i in tqdm.tqdm(points, f"Level {n}"):
            for j in points:
                for k in points:
                    if mask[i,j,k] == 0:
                        z = np.quaternion(0,i+1/2, j+1/2, k+1/2)*resolution + origin 
                        d = distance_function(z,**kwargs)
                        r = int(np.floor(np.abs(d)/np.sqrt(3)/resolution)-0.5)
                        if r >= span:
                            mask[i-span:i+span+1,j-span:j+span+1,k-span:k+span+1] = np.sign(d)
    return mask <= cutoff

def iterate_voxels_simple(distance_function, levels, size=1,cutoff = 0, kwargs = {}):
    steps = 3**levels
    points = np.linspace(-size/2, size/2, steps)
    X, Y, Z = np.meshgrid(points, points, points)
    Q = quaternion.from_float_array(np.array([np.zeros_like(X), X, Y, Z]).transpose([1,2,3,0]))
    d = distance_function(Q, **kwargs)
    return d < cutoff


def export_stl(path, verts, faces, header = None):
    """Export an STL based on a set of vertecies and faces, with
    and optional header. Written by GPT. 

    Args:
        path (str | pathlib.Path): String or path object specifying output file. 
        verts ((V, 3) array): Array specifying the coordinates of each vertex 
        faces ((F, 3) array): Array specifying the vertecies of each (triangular) face
        header (str, optional): Header to add to stl. Defaults to None. Must be 
            less than 80 characters. 

    Raises:
        ValueError: If the header is longer than 80 characters.
    """
    if header is None:
        header = b' ' * 80
    elif len(header) > 80:
        raise ValueError("Length of header must not exceed 80 bytes!")
    else:
        header = header + b' '*(80-len(header))
    
    tri_count = faces.shape[0]
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", tri_count))
        for i0, i1, i2 in tqdm.tqdm(faces.astype(np.uint32),desc="Exporting Faces"):
            a, b, c = verts[i0], verts[i1], verts[i2]
            # Compute flat normal per triangle
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm == 0.0:
                # Skip degenerate triangle
                continue
            n = n / (norm if norm > 0 else 1.0)
            f.write(struct.pack("<fff", *n.astype(np.float32)))
            f.write(struct.pack("<fff", *a))
            f.write(struct.pack("<fff", *b))
            f.write(struct.pack("<fff", *c))
            f.write(struct.pack("<H", 0))
