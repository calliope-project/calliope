.. _installation:

=========================
Download and installation
=========================

Requirements
============

Calliope has been tested on Linux, macOS, and Windows.

Running Calliope requires four things:

1. The Python programming language, version 3.9 to 3.11.
2. A number of Python add-on modules (see :ref:`below for the complete list <python_module_requirements>`).
3. A solver: Calliope has been tested with CBC, GLPK, Gurobi, and CPLEX. Any other solver that is compatible with Pyomo should also work.
4. The Calliope software itself.

Recommended installation method
===============================

The easiest way to get a working Calliope installation is to use the free ``mamba`` package manager, which can install all of the four things described above in a single step.

To get ``mamba``, `download and install the "Mambaforge" distribution for your operating system <https://mamba.readthedocs.io/en/latest/index.html>`_ (using the version for Python 3).

With mamba installed, you can create a new environment called ``calliope`` with all the necessary modules, including the free and open source GLPK solver, by running the following command in a terminal or command-line window

  .. code-block:: fishshell

    $ mamba create -c conda-forge -n calliope calliope

This will install calliope with Python version 3.11.

To use Calliope, you need to activate the ``calliope`` environment each time

  .. code-block:: fishshell

    $ mamba activate calliope

You are now ready to use Calliope together with the free and open source GLPK solver.
However, we recommend to not use this solver where possible, since it performs relatively poorly (both in solution time and stability of result).
Indeed, our example models use the free and open source CBC solver instead, but installing it on Windows requires an extra step. Read the next section for more information on installing alternative solvers.

.. warning::

    Although possible, we do not recommend installing Calliope directly via ``pip`` (``pip install calliope``).
    Non-python binaries are not installed with ``pip``, some of which are necessary for stable operation (e.g., `libnetcdf`).

Updating an existing installation
=================================

If following the recommended installation method above, the following command, assuming the mamba environment is active, will update Calliope to the newest version

  .. code-block:: fishshell

    $ mamba update -c conda-forge calliope

.. _install_solvers:

Solvers
=======

You need at least one of the solvers supported by Pyomo installed. CBC (open-source) or Gurobi (commercial) are recommended for large problems, and have been confirmed to work with Calliope. Refer to the documentation of your solver on how to install it.

CBC
---

`CBC <https://github.com/coin-or/Cbc>`_ is our recommended option if you want a free and open-source solver. CBC can be installed via conda on Linux and macOS by running ```mamba install -c conda-forge coincbc```. Windows binary packages are somewhat more difficult to install, due to limited information on `the CBC website <https://github.com/coin-or/Cbc>`_, but can be found within their `binary archive <https://www.coin-or.org/download/binary/Cbc/>`_ and are included in their `package releases on GitHub <https://github.com/coin-or/Cbc/releases>`_. The GitHub releases are more up-to-date. We recommend you download the relevant binary for `CBC 2.10.8 <https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.8/Cbc-releases.2.10.8-w64-msvc17-md.zip>`_ and add `cbc.exe` to a directory known to PATH (e.g. an Anaconda environment 'bin' directory).

GLPK
----

`GLPK <https://www.gnu.org/software/glpk/>`_ is free and open-source, but can take too much time and/or too much memory on larger problems. If using the recommended installation approach above, GLPK is already installed in the ``calliope`` environment. To install GLPK manually, refer to the `GLPK website <https://www.gnu.org/software/glpk/>`_.

Gurobi
------

`Gurobi <https://www.gurobi.com/>`_ is commercial but significantly faster than CBC and GLPK, which is relevant for larger problems. It needs a license to work, which can be obtained for free for academic use by creating an account on gurobi.com.

While Gurobi can be installed via conda (:sh:`mamba install -c gurobi gurobi`) we recommend downloading and installing the installer from the `Gurobi website <https://www.gurobi.com/>`_, as the conda package has repeatedly shown various issues.

After installing, log on to the `Gurobi website <https://www.gurobi.com/>`_ and obtain a (free academic or paid commercial) license, then activate it on your system via the instructions given online (using the ``grbgetkey`` command).

CPLEX
-----

Another commercial alternative is `CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_. IBM offer academic licenses for CPLEX. Refer to the IBM website for details.

.. _python_module_requirements:

Python module requirements
==========================

Refer to `requirements.txt <https://github.com/calliope-project/calliope/blob/main/requirements.txt>`_ in the Calliope repository for a full and up-to-date listing of required third-party packages.

Some of the key packages Calliope relies on are:

* `Pyomo <https://www.pyomo.org/>`_
* `Pandas <https://pandas.pydata.org/>`_
* `Xarray <https://docs.xarray.dev/en/stable/>`_
* `Jupyter <https://jupyter.org/>`_ (optional, but highly recommended, and used for the example notebooks in the tutorials)
