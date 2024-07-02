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

The `self.probe_task()` can be called inside the `_compute()` method, no need to do it externally in `compute_and_update()`.
If you override the latter, make sure to update the control input of the agent, as the super class does.

```py
def _compute(self):
    # ...
    zNew, vNew, riskNew, flag = self.probe_task()
    # ...
```

The `_compute()` method is supposed to return two values: a numpy array with the control input and a time series, which can also be empty.
In the case of the pybullet racecar, the control input is a pair containing the velocity and the steering angle.

```py
def _compute(self):
    # ...
    return np.array([speed, steering_angle]), TimeSeries()
```

The reason for the "jumping" cars is because you instantiate them into the ground, so they are pushed up by the physics engine.
Furthermore, since you do not provide any control input, they won't not move on their own.

```py
# height = 0.5 instead of 0
initial_states = [[5, 5, 0.5], [15, 5, 0.5], [25, 5, 0.5], [35, 5, 0.5]]
```

## Tamas imports

I encountered some trouble importing the `tamas` package.
More specifically, it seems that some imports are relative (e.g. `from .<name> import <name2>`) and some are absolute (e.g. `from <name> import <name2>`)
The latter kind of imports was not working properly.
I fixed the issue by making all imports relative.
Maybe double check this on your end.
