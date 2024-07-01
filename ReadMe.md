# Task Allocation for Stochastic Multi-Agent Systems (TASMAS) with SymAware framework

**Author:** Zengjie Zhang (z.zhang3@tue.nl)

This project demonstrates the risk-aware allocation of real-time STL specifications for stochastic multi-agent systems, with restricted risk probabilities. 

## Introduction

### Associated Research Work

This library is associated with the Arxiv article in [https://arxiv.org/abs/2404.02111](https://arxiv.org/abs/2404.02111).

### Relation with Existing Toolbox

The `probstlpy` library in this project is modified from the [stlpy](https://github.com/vincekurtz/stlpy/blob/main/README.md) toolbox. 

### Scenario

This library considers a multi-bus routing scenario in a tourism attraction point to validate the effectiveness of the proposed method. The scenario contains four bus terminals denoted as **T-X**, where **X**$\in \{A, B, C, D\}$, four tourist gathering points denoted as **GP-Z**, where **Z**$\in \{I, II, III, IV\}$, and a single unloading point denoted as **ULP** (see the following Figure). Four buses, $A$ to $D$, are tasked with picking up tourists from the gathering points and transporting them to the unloading point. These buses initiate operations from their respective terminals and are expected to return when required. All buses are confined within the attraction point symbolized as ‘**BOX**’. The buses must avoid running into two buildings denoted as **B-Y**, with **Y**$\in \{1, 2\}$. When the tourists at a gathering point reach a certain number, a bus should be available to transport them to the unloading point within a tolerable time limit. Therefore, this study needs to handle dynamically allocated routing tasks, implying that new routing tasks may be assigned at any time $k \in \{0, 1,· · · , N − 1\}$.

[![Map](map.svg)](CASE)





## Installation

### System Requirements

**Operating system**
 - *Windows* (compatible in general, succeed on 11)

**Python Environment**
 - Python version: test passed on `python=3.11`
 - **Recommended**: IDE ([VS code](https://code.visualstudio.com/) or [Pycharm](https://www.jetbrains.com/pycharm/)) and [Conda](https://www.anaconda.com/)
 - Required Packages: `numpy`, `treelib`, `matplotlib`, `scipy`, `pybullet`. 
 
**C/C++ Building Tool**
 - *Microsoft Visual C++* (14.0 or greater). Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). In the installer, select the `C++ build tools` workload and ensure that the following components are checked:
    - MSVC v142 - VS 2019 C++ x64/x86 build tools (Latest)
    - Windows 10 SDK (Latest)
    - C++ CMake tools for Windows

 **Required Libraries**
 - `gurobipy` solver (**license** required, see [How to Get a Gurobi License](https://www.gurobi.com/solutions/licensing/))
 - `Python control` toolbox (see [Documentation](https://python-control.readthedocs.io/en/latest/intro.html))
 
### Configuring Python Environment
 
1. Install conda following this [instruction](https://conda.io/projects/conda/en/latest/user-guide/install/index.html);

2. Open the conda shell, and create an independent project environment;
```
conda create --name tasmas-symaware python=3.11
```

3. In the same shell, activate the created environment
```
conda activate tasmas-symaware
```

4. In the same shell, within the `tasmas` environment, install the dependencies one by one
 ```
conda install -c anaconda numpy
conda install -c conda-forge treelib
conda install -c conda-forge matplotlib
conda install -c anaconda scipy
```

5. In the same shell, within the `tasmas` environment, install the libraries
```
python -m pip install gurobipy
pip install control
pip install pybullet
```

6. Last but not least, activate the `gurobi` license (See [How To](https://www.gurobi.com/documentation/current/remoteservices/licensing.html)). Note that this project is compatible with `gurobi` Released version `11.0.1`. Keep your `gurobi` updated in case of incompatibility. 

### Configure Git repositories

1. Clone this repository:
```
git clone git@github.com:zhang-zengjie/tasmas-symaware-base.git
```

2. Clone the [tasmas](https://github.com/zhang-zengjie/tasmas) repository and the `eicsymaware` framework
```
cd tasmas-symaware-base
git clone git@github.com:zhang-zengjie/tasmas.git
```

3. Download the [EICSymAware-base](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/archive/base/eicsymaware-base.zip) zip file. Unzip the file and place the folder `eicsymaware-base/src/symaware` under the `tasmas-symaware-base` directory.

4. Download the [EICSymAware-pybullet](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/archive/pybullet/eicsymaware-pybullet.zip) zip file. Unzip the file and place the folder `eicsymaware-pybullet/src/symaware/simulators` under the `tasmas-symaware-base/symaware` directory.


### Running Instructions

- Run the main script `main.py`;
- Watch the terminal for runtime information;
- The pybullet simulation environment will prompt up automatically.

## License

This project is with a BSD-3 license, refer to `LICENSE` for details.
