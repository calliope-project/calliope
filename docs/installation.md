# Download and installation

## Requirements

Calliope is tested on Linux, macOS, and Windows.

Running Calliope requires four things:

1. The Python programming language, version {{ min_python_version }} to {{ max_python_version }}.
2. A number of Python add-on modules including [Pyomo](https://www.pyomo.org/), [Pandas](https://pandas.pydata.org/) and [Xarray](https://xarray.dev/).
3. An optimisation solver: Calliope has been tested with CBC, GLPK, Gurobi, and CPLEX. Any other solver that is compatible with Pyomo should also work.
4. The Calliope software itself.

## Recommended installation method

The easiest way to get a working Calliope installation is to use the free `mamba` package manager, which can install all of the four things described above in a single step.

To get `mamba`, the most straightforward approach is to [download and install the "Miniforge" distribution for your operating system](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

!!! tip

    Another option is to use the commercially developed [Anaconda Python distribution](https://www.anaconda.com/download), which is available for all operating systems and comes with a graphical user interface to install and manage packages.
    If you use the Anaconda distribution, you have to replace `mamba` with `conda` in the installation instructions below.

With the package manager installed, you can create a new environment called `calliope` with Calliope as well as the free and open source GLPK solver.
Run the following command in a terminal or command-line window:

```shell
mamba create -n calliope -c conda-forge/label/calliope_dev -c conda-forge calliope glpk
```

!!! note
    The `conda-forge/label/calliope_dev` channel allows you to access the pre-release of Calliope v0.7, with which this version of the documentation aligns.
    To install the most recent _stable_ version of Calliope, see our [v0.6.10 documentation](https://calliope.readthedocs.io/en/v0.6.10/).

This will install calliope with Python version {{ max_python_version }}.

To use Calliope, you need to activate the `calliope` environment each time

```bash
mamba activate calliope
```

!!! warning

    Although possible, we do not recommend installing Calliope directly via `pip` (`pip install calliope`).
    Non-python binaries are not installed with `pip`, some of which are necessary for stable operation (e.g., `libnetcdf`).

## Updating an existing installation

If following the recommended installation method above, the following command, assuming the mamba environment is active, will update Calliope to the newest version

```bash
mamba update -c conda-forge calliope
```

## Choosing a solver

You cannot solve a Calliope model until you have installed a solver.
The easiest solver to install is [GLPK](#glpk).
However, we recommend to not use this solver where possible, since it performs relatively poorly (both in solution time and stability of result).
Indeed, our example models use the free and open source [CBC](#cbc) solver instead, but installing it on Windows requires an extra step.
[CBC](#cbc) (open-source) or [Gurobi](#gurobi) (commercial) are recommended for large problems, and have been confirmed to work with Calliope.
The following subsections provide additional detail on how to install a solver.
This list is not exhaustive; any solvers [supported by Pyomo](https://pyomo.readthedocs.io/en/latest/reference/topical/appsi/appsi.solvers.html) can be used.

### CBC

[CBC](https://github.com/coin-or/Cbc) is our recommended option if you want a free and open-source solver.
CBC can be installed via conda on Linux and macOS by running `mamba install -c conda-forge coin-or-cbc`.
Windows binary packages are somewhat more difficult to install, due to limited information on [the CBC website](https://github.com/coin-or/Cbc), but are included in their [package releases on GitHub](https://github.com/coin-or/Cbc/releases).
The GitHub releases are more up-to-date. We recommend you download the relevant binary for [CBC 2.10.11](https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.11/Cbc-releases.2.10.11-w64-msvc17-md.zip) and add `cbc.exe` to a directory known to PATH (e.g. an Anaconda environment 'bin' directory).

### GLPK

[GLPK](https://www.gnu.org/software/glpk/) is free and open-source, but can take too much time and/or too much memory on larger problems.
`GLPK` can be installed from `conda-forge` on all platforms: `mamba install glpk`.
It is therefore the _easiest_ solver to have installed in your Calliope environment.

### Gurobi

[Gurobi](https://www.gurobi.com/) is commercial but significantly faster than CBC and GLPK, which is relevant for larger problems.
It needs a license to work, which [can be obtained for free for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/).

The Gurobi solver interface can be installed via conda (`mamba install gurobi::gurobi`).

After installing, log on to the [Gurobi website](https://www.gurobi.com/) and obtain a (free academic or paid commercial) license, then activate it on your system via the instructions given online (using the `grbgetkey` command).

!!! note
    If using the Gurobi solver, you can also leverage the reduced time and memory consumption of our [Gurobi optimisation problem backend](advanced/backend_choice.md) - this circumvents Pyomo entirely.

### CPLEX

Another commercial alternative is [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio).
IBM offer academic licenses for CPLEX. Refer to the IBM website for details.
