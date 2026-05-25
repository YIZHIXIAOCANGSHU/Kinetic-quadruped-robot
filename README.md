# Go2 Simulation

Traditional dynamics-based Unitree Go2 simulation with MuJoCo, MPC/WBC control,
depth-camera perception, and Rerun diagnostics.

## Project Layout

- `src/go2_simulation/`: executable simulation, perception, sensors, and Rerun logging.
- `src/convex_mpc/`: runtime MPC/WBC controller package.
- `models/`: runtime MJCF/URDF assets used by the simulator.
- `tests/`: unit and smoke tests for control/perception glue.
- `legacy/`: reference material, original examples, media, and unused official model copies.

## Run

The root scripts use a project-local `.venv`. If it does not exist,
`activate_go2_venv.sh` creates it with Python 3.10 and installs the required
pip packages from `requirements.txt`.

```bash
./run_go2_sim.sh
```

For an interactive shell:

```bash
source ./activate_go2_venv.sh
```

If your Python 3.10 binary has a different name:

```bash
PYTHON_BIN=/path/to/python3.10 source ./activate_go2_venv.sh
```

Useful runtime flags:

```bash
GO2_DEPTH_OBSTACLE=0 ./run_go2_sim.sh
GO2_ENABLE_LIDAR=1 ./run_go2_sim.sh
GO2_RERUN_RECORD_PATH=go2.rrd ./run_go2_sim.sh
```

LiDAR is disabled by default. The default path uses the official front depth
camera and Rerun diagnostics.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests

PYTHONPYCACHEPREFIX=/tmp/go2_pycache .venv/bin/python -m py_compile \
  simulate_go2.py src/go2_simulation/*.py src/go2_simulation/*/*.py src/convex_mpc/*.py
```
