.. _installation:

=========================
Download and installation
=========================

Requirements
============

Calliope has been tested on Linux, macOS, and Windows.

Running Calliope requires four things:

1. The Python programming language, version 3.6 or higher.
2. A number of Python add-on modules (see :ref:`below for the complete list <python_module_requirements>`).
3. A solver: Calliope has been tested with GLPK, CBC, Gurobi, and CPLEX. Any other solver that is compatible with Pyomo should also work.
4. The Calliope software itself.

Recommended installation method
===============================

The easiest way to get a working Calliope installation is to use the free ``conda``package manager, which can install all of the four things described above in a single step.

To get ``conda``, `download and install the "Miniconda" distribution for your operating system <https://conda.io/miniconda.html>`_ (using the version for Python 3).

With Miniconda installed, you can create a new Python 3.6 environment called ``"calliope"`` with all the necessary modules, including the free and open source GLPK solver, by running the following command in a terminal or command-line window::

   $ conda create -c conda-forge -n calliope python=3.6 calliope

To use Calliope, you need to activate the ``calliope`` environment each time. On Linux and macOS::

   $ source activate calliope

On Windows::

   $ activate calliope

You are now ready to use Calliope together with the free and open source GLPK solver. Read the next section for more information on alternative solvers.

Solvers
=======

You need at least one of the solvers supported by Pyomo installed. CPLEX or Gurobi are recommended for large problems, and have been confirmed to work with Calliope. Refer to the documentation of your solver on how to install it.

GLPK
----

`GLPK <https://www.gnu.org/software/glpk/>`_ is free and open-source, but can take too much time and/or too much memory on larger problems. If using the recommended installation approach  above, GLPK is already installed in the ``calliope`` environment. To install GLPK manually, refer to the `GLPK website <https://www.gnu.org/software/glpk/>`_.

CBC
---

`CBC <https://projects.coin-or.org/Cbc>`_ is another free and open-source option. CBC can be installed via conda on Linux and macOS by running ``conda install -c conda-forge coincbc``. Windows binary packages and further documentation are available at `the CBC website <https://projects.coin-or.org/Cbc>`_.

Gurobi
------

`Gurobi <https://www.gurobi.com/>`_ is commercial but significantly faster than GLPK and CBC, which is relevant for larger problems. It needs a license to work, which can be obtained for free for academic use by creating an account on gurobi.com.

While Gurobi can be installed via conda (``conda install -c gurobi gurobi``) we recommend downloading and installing the installer from the `Gurobi website <https://www.gurobi.com/>`_, as the conda package has repeatedly shown various issues.

After installing, log on to the `Gurobi website <https://www.gurobi.com/>`_ and obtain a (free academic or paid commercial) license, then activate it on your system via the instructions given online (using the ``grbgetkey`` command).

CPLEX
-----

Another commercial alternative is `CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_. IBM offer academic licenses for CPLEX. Refer to the IBM website for details.

.. _python_module_requirements:

Python module requirements
==========================

The following Python modules and their dependencies are required:

* `Pyomo <https://software.sandia.gov/trac/pyomo/wiki/Pyomo>`_
* `Pandas <http://pandas.pydata.org/>`_
* `Xarray <http://xarray.pydata.org/>`_
* `NetCDF4 <https://github.com/Unidata/netcdf4-python>`_
* `Numexpr <https://github.com/pydata/numexpr>`_
* `ruamel.yaml <https://yaml.readthedocs.io/en/latest/>`_
* `Click <http://click.pocoo.org/>`_

`Plotly <https://plot.ly/>`_ is optional but necessary to graphically display results.

`Jupyter <http://jupyter.org/>`_ is optional and used for the example notebook in the tutorial.
