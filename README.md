# fractal-printer
Python scripts for generating 3D Julia set STL files for 3D printing. 

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it.
2. Clone the repo and install dependencies:
   ```bash
   git clone <this-repo-url>
   cd fractal-printer
   uv sync
   ```
   This creates a `.venv` in the project directory with Python 3.11+ and all dependencies (including `sdf`, pulled directly from [fogleman/sdf](https://github.com/fogleman/sdf) on GitHub) pinned to the versions in `uv.lock`.
3. Run scripts and notebooks through uv so they use that environment, e.g.:
   ```bash
   uv run python scripts/main.py -i inputs/example_julia.json -o outputs/example.stl
   uv run jupyter lab
   ```
   Or activate the environment directly: `.venv\Scripts\activate` (Windows) / `source .venv/bin/activate` (macOS/Linux).

Note: `PyQt6` (used for the interactive preview window) is licensed under GPLv3 unless you hold a commercial Qt license.

Wish list:
1. Add options for multi-order polynomials?!?
2. Make some art.

