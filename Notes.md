# Notes

## Installation

All external pip packages can be installed with

```bash
pip install -r requirements.txt --extra-index-url https://gitlab.mpi-sws.org/api/v4/projects/2668/packages/pypi/simple
```

This includes both `symaware-base` and `symaware-pybullet`.
No need to clone the Gitlab repository.

The only exception is the `slycot` package, indicated by the `Python control` toolbox, which was a bit tricky to install, so maybe it is better to leave it to the user.
The installation can still happen globally, in the conda environment, or in a virtual environment.

The user still needs to clone the tamas repository.

## Source code changes

No need of changing the path to accomodate the `symaware` packages, since now they are installed as pip packages.

```py
# Removed
root_path = os.getcwd()
sys.path.append(os.path.join(root_path, 'eicsymaware', 'src'))
```

