import json
import tqdm
import fractal_printer as fp
from skimage.measure import marching_cubes
from pathlib import Path
from datetime import datetime
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import sys
import subprocess

def make_solid(config, output, visualize = False, simple_iteration=False):
    
    if "distance_function" not in config:
        raise ValueError("Config must contain a distance_function entry!")
    if config["distance_function"] not in fp.distance_functions.functions:
        raise ValueError(f"Specified distance function ({config["distance_function"]}) not in library!\nOptions are {fp.distance_functions.functions.keys()}")

    # Add default settings
    if "levels" not in config:
        config["levels"] = 5

    if "size" not in config:
        config["size"] = 2

    if "arguments" not in config:
        config["arguments"] = None
    
    print("Iterating voxels...")
    if simple_iteration:
        mask = fp.iterate_voxels_simple(
            fp.distance_functions.functions[config["distance_function"]],
            levels = config["levels"], 
            size = config["size"],
            kwargs = config["arguments"]
        )
    else:
        mask = fp.iterate_voxels(
            fp.distance_functions.functions[config["distance_function"]],
            levels = config["levels"], 
            size = config["size"],
            kwargs = config["arguments"]
        )

    print("Finding Faces")
    verts, faces, _normals, _values = marching_cubes(volume = mask, **config["mesh_arguments"])

    print("Exporting")
    fp.export_stl(output, verts, faces)  

    if visualize:
        print("Plotting")
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_facecolor("lightblue")
        mesh.set_edgecolor("k")
        ax.add_collection3d(mesh)

def main(argv):
    parser = argparse.ArgumentParser(
                    prog='Fractal Printer',
                    description='This script generates stl files representing 3D shapes generated from a provided distance function')
    
    parser.add_argument('filename') 
    parser.add_argument("-i","--input")
    parser.add_argument("-o","--output",
                        default=Path(os.getcwd()) / "outputs" / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
    )
    parser.add_argument("-v","--visualize",action="store_true")
    parser.add_argument("--open",action="store_true")
    parser.add_argument("--use_simple_iteration",action="store_true")
    args = parser.parse_args(argv)

    with open(args.input) as f:
        config = json.load(f)

    make_solid(config, args.output, visualize=args.visualize, simple_iteration = args.use_simple_iteration)

    if args.open:
        subprocess.run(["open",args.output])
    
if __name__ == "__main__":
    main(sys.argv)
    