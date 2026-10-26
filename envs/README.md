# Environments

The core package supports Python 3.10+, but the locked LeRobot 0.6.2 source
requires **Python 3.12+**. Use a dedicated Python 3.12 training environment;
do not substitute an older installed LeRobot just because its import succeeds.
The historical tactile simulation worker is a separate environment, invoked
through the configured worker Python executable.

From the repository root, after creating/activating a clean Python 3.12
environment, reproduce the tested Linux CPU environment:

```bash
unset PYTHONPATH  # Do not inherit packages from ROS or another environment.
python scripts/bootstrap_dependencies.py --dependency lerobot --fetch
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r envs/cpu-py312.lock.txt
python -m pip install --no-deps --no-build-isolation -e '.[train,report,dev]' -e external/lerobot
python -m pip check
python scripts/doctor.py --profile train
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

`--local lerobot=/path/to/checkout --local vggt=/path/to/checkout` reuses existing
clean checkouts at the locked commits without downloading them. Set
`TACTILE_VGGT_DEPS_ROOT` or `--shared-root` to place the shared pool on a data
disk. The ignored links are recreated per checkout; machine paths are not
part of the source lock.

`cpu-py312.lock.txt` records the isolated CPU environment's installed dependency
set, including transitive versions. The two editable projects are installed
from their locked source checkouts, so this file contains no machine paths.
It is Linux/Python 3.12/CPU specific, not a GPU training lock.
MuJoCo 3.4.0 is included to run the synthetic simulator-state regression test;
passing it does not validate the private tactile model, robot assets, or full
closed-loop task. Those prerequisites remain separate doctor checks.

For GPU training, select compatible official PyTorch CUDA wheels for your
host and use `core-constraints.txt` when installing `.[train,report,dev]` and
the pinned `external/lerobot`. Those direct constraints are not an exhaustive
transitive lock. Record `python -m pip freeze` together with
the Python/CUDA versions in each run manifest or run environment file.
Do not export an environment's local editable paths as a portable lock file.

For dependency-free repository tooling or NumPy geometry checks, install
`python -m pip install -c envs/core-constraints.txt -e '.[dev]'` and use
`python scripts/doctor.py --profile core`. The full test suite additionally
needs the train environment; Torch-dependent tests may skip when Torch is
absent. A core check is not an ACT, VGGT-weight, or simulator validation.

For simulation, use `simulation-constraints.txt` with the frozen external
tactile environment. The project does not download or redistribute private
demonstrations, robot assets, or historical tactile readout code. The doctor
reports those missing prerequisites explicitly.
