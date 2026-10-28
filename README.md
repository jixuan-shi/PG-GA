# PG-GA

Source code for **Joint Intersection-Free Task Assignment and Motion Planning for Workload-Balanced Multi-UAV IoT Data Collection**.

## Methods

- Smart agriculture: the **Polar-Guided Genetic Algorithm (PG-GA)** for task assignment in an obstacle-free environment.
- Post-disaster: the **cooperative hierarchical framework**, combining task assignment, a three-stage motion planner, and dynamic cost feedback. The planner uses adaptive-step RRT, greedy path simplification, and local optimization.

## Installation

Use Python 3.12 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The elevation raster is managed with Git LFS. After cloning, retrieve it with:

```bash
git lfs install
git lfs pull
```

## Run

From the repository directory:

```bash
python run_smart_agriculture.py
python run_post_disaster.py
```

The launchers select the appropriate working directory and module search paths. Figures are written to the corresponding scenario directory. Algorithm parameters and task-generation settings are defined in each `our_proposed.py` script.

## Files

```text
SmartA/ours/       PG-GA task assignment and Agriculture.png
PostD/ours/        Cooperative framework and city2.tif
map/              DEM processing
mtsp/             Task-assignment genetic operators
path_planning/    RRT, path simplification, and local optimization
VrpLib/           eil51, kroA100, kroB150, kroA200, and pr226
```

`VrpLib/` contains the original benchmark files used in the project. In the agriculture script, random DCP generation is active by default. To use a benchmark, enable the `load_tsp_dataset(file_path)` call and select the desired file, number of DCPs, and generation budget.

The original algorithm scripts and dependencies are preserved. The agriculture dataset path is made repository-relative; the launchers provide portable entry points.
