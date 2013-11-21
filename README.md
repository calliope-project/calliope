# Calliope: a multi-scale energy systems (MUSES) model

Work in progress.

## Requirements

Calliope is only tested on Python 2.7.x and unlikely to work on older versions. Support for 3.x will be implemented as soon as all required modules support Python 3.

The following Python modules are required:

* Coopr (Pyomo)
* NumPy
* Pandas
* PyYAML

In addition, Pyomo requires a solver. The default (hard-coded) configuration uses IBM ILOG CPLEX, but only a few lines of code need changing to use other solvers. A free/open-source alternative is [GLPK](https://www.gnu.org/software/glpk/). GLPK is known to be memory-intensive and therefore possibly unsuitable for large problems.

The recommended way to obtain all required Python modules with the exception of Coopr/Pyomo is to use the [Anaconda distribution](https://store.continuum.io/cshop/anaconda/). Once that is set up and configured as the active Python interpreter, install Coopr/Pyomo with `pip install coopr`.

## Installation

To install the latest stable version via `pip`:

```
pip install -e git+https://github.com/sjpfenninger/calliope.git@master#egg=calliope
```

## Use

### Running on a Sun Grid Engine or compatible cluster (`qsub`)

* Create a configuration in `runs_configs`, based on `calliope/parallel_settings.yaml`.
* Run `model_run.py runs_configs/your_settings.yaml`.
* If not running on the cluster controller machine already, copy the resulting folder from `runs` to the cluster.
* On the cluster controller, `cd` into the runs folder and submit the jobs by `qsub run.sh`.

Other supported ways to run the model are stand-alone runs (locally e.g. from within an IPython notebook) and Platform LSF clusters (`bsub`).
