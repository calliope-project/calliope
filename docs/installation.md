# Download and installation

## Requirements

Calliope is tested on Linux, macOS, and Windows.

Running Calliope requires four things:

1. The Python programming language, version {{ min_python_version }} to {{ max_python_version }}.
2. A number of Python add-on modules including [Pyomo](https://www.pyomo.org/), [Pandas](https://pandas.pydata.org/) and [Xarray](https://docs.xarray.dev/).
3. An optimisation solver: Calliope has been tested with CBC, GLPK, and Gurobi.
   Any other solver that is compatible with Pyomo should also work.
4. The Calliope software itself.

## Recommended installation methods

We recommend two installation paths depending on your use-case:

1. **As a user with `conda` or `pixi` (recommended)**: use conda or [pixi](https://pixi.sh/latest/).
   This is the recommended option when you want Calliope plus solver and binary dependencies managed together.
1. **As a user with `uv` or `pip`**: use pip or [`uv`](https://docs.astral.sh/uv/) for fast Python package installation.
   You will need to ensure you have all non-Python libraries installed and available if taking this approach.

### As a User with `conda` or `pixi` (recommended)

=== "`pixi`"
    Install `pixi` by following the [official installation instructions](https://pixi.sh/latest/installation/).
    Then add `calliope` to your project workspace:

    ```shell
    pixi add conda-forge::calliope
    ```

    !!! note
        If you cannot directly install `pixi` due to organisational restrictions, you can also install it in a `conda` environment and use it from there:

        ```shell
        conda install pixi
        ```

=== "`conda`"
    Install `conda` by following the [official installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
    Then, create an environment including `calliope`:

    ```shell
    conda create -n calliope conda-forge::calliope
    ```

To run commands in the environment, use `pixi run -e dev ...` (or `pixi shell -e dev`).

### As a user with `uv` or `pip`

If you only need to install Calliope from PyPI, use:

=== "`uv`"
    ```shell
    uv pip install calliope
    ```

=== "`pip`"
    ```shell
    pip install calliope
    ```

!!! warning

    Although possible, we do not recommend installing Calliope directly via `pip` or `uv`.
    Non-python binaries are not installed with `pip`, some of which are necessary for stable operation (e.g., `libnetcdf`).

## Choosing a solver

You cannot solve a Calliope model until you have installed a solver.
The easiest solver to install is [CBC](#cbc), which is included if you follow the [recommended installation instructions](#as-a-user-with-conda-or-pixi-recommended) above.
[CBC](#cbc) (open-source) or [Gurobi](#gurobi) (commercial) are recommended for large problems, and have been confirmed to work with Calliope.
The following subsections provide additional detail on how to install a solver.
This list is not exhaustive; any solvers [supported by Pyomo](https://pyomo.readthedocs.io/en/latest/reference/topical/appsi/appsi.solvers.html) can be used.

!!! note

    The HiGHS solver _is not supported_ by our Pyomo backend.
    This is due to the HiGHS interface not being supported by the Pyomo kernel interface, which we use.

### CBC

[CBC](https://github.com/coin-or/Cbc) is our recommended option if you want a free and open-source solver.
If you do not have it in your working environment (i.e. there is nothing listed when you call `conda list cbc`/`pixi list cbc`) then it can be installed on all platforms:

=== "`pixi`"

    ```shell
    pixi add conda-forge::coin-or-cbc
    ```

=== "`conda`"

    ```shell
    conda install conda-forge::coin-or-cbc
    ```

### GLPK

[GLPK](https://anaconda.org/conda-forge/glpk) is free and open-source, but can take too much time and/or too much memory on larger problems.
`GLPK` can be installed from `conda-forge` on all platforms:

=== "`pixi`"

    ```shell
    pixi add conda-forge::glpk
    ```

=== "`conda`"

    ```shell
    conda install conda-forge::glpk
    ```

Unlike [CBC](#cbc), it is possible to extract [shadow prices](./advanced/shadow_prices.md) from a model solved with GLPK, which is why you may with to use it instead of CBC.

### Gurobi

[Gurobi](https://www.gurobi.com/) is commercial but significantly faster than CBC and GLPK, which is relevant for larger problems.
It needs a license to work, which [can be obtained for free for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/).

=== "`pixi`"

    

After installing, log on to the [Gurobi website](https://www.gurobi.com/) and obtain a (free academic or paid commercial) license, then activate it on your system via the instructions given online (using the `grbgetkey` command).

!!! note
    If using the Gurobi solver, you can also leverage the reduced time and memory consumption of our [Gurobi optimisation problem backend](advanced/backend_choice.md) - this circumvents Pyomo entirely.

### CPLEX

Another commercial alternative is [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio).
IBM offers academic licenses for CPLEX. Refer to the IBM website for details.

!!! tip
    After installing CPLEX, it is important to ensure that the path to the solver is part of the environment variables, which is typically not automatic. Please follow the steps given for your operating system on CPLEX's [dedicated documentation](https://www.ibm.com/docs/en/icos/22.1.0?topic=cplex-setting-up).

## Customising the solver's performance

Solvers typically allow users to specify custom `solver_options`, by which you may tailor their performance to what best suits the features of the model you are working with. For further information, see our guide on [solver options customisation ](advanced/solver.md).
