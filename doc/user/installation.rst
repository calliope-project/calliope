.. _installation:

=========================
Download and installation
=========================

Requirements
============

Calliope has been tested on Linux, macOS, and Windows.

Running Calliope requires four things:

1. The Python programming language, version 3.5 or higher.
2. A number of Python add-on modules (see :ref:`below for the complete list <python_module_requirements>`).
3. A solver: Calliope has been tested with `GLPK <https://www.gnu.org/software/glpk/>`_, `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_, and `Gurobi <http://www.gurobi.com/>`_. Any other solver that is compatible with Pyomo, which Calliope uses to construct its models, should work.
4. The Calliope software itself.


Recommended installation method
===============================

The easiest way to get a working Calliope installation is to use the free `Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_ and its package manager, ``conda``.

With Anaconda installed, you can create a new Python 3.5 environment called "calliope" with all the necessary modules, including the free and open source GLPK solver, with the following command::

   $ conda create -c conda-forge -n calliope python=3.5 calliope

To use Calliope, you need to activate the "calliope" environment each time. On Linux and macOS::

   $ source activate calliope

On Windows::

   $ activate calliope

You are now ready to use Calliope together with the free and open source GLPK solver. Read the next section for more information on alternative solvers.

Solvers
=======

You need at least one of the solvers supported by Pyomo installed. GLPK or Gurobi are recommended and have been confirmed to work with Calliope. Refer to the documentation of your solver on how to install it. Some details on GLPK and Gurobi are given below. Another commercial alternative is `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_.

GLPK
----

`GLPK <https://www.gnu.org/software/glpk/>`_ is free and open-source, but can take too much time and/or too much memory on larger problems. If using the recommended installation approach  above, GLPK is already installed in the "calliope" environment. To install GLPK manually, refer to the `GLPK website <https://www.gnu.org/software/glpk/>`_.

Gurobi
------

`Gurobi <http://www.gurobi.com/>`_ is commercial but significantly faster than GLPK, which is relevant for larger problems. It needs a license to work, which can be obtained for free for academic use by creating an account on gurobi.com.

Like Calliope itself, Gurobi can also be installed via conda::

    $ conda install -c gurobi gurobi

After installing, log on to the `Gurobi website <http://www.gurobi.com/>`_ and obtain a (free academic or paid commercial) license, then activate it on your system via the instructions given online (using the ``grbgetkey`` command).

.. _python_module_requirements:

Python module requirements
==========================

The following Python modules and their dependencies are required:

* `Pyomo <https://software.sandia.gov/trac/pyomo/wiki/Pyomo>`_
* `Pandas <http://pandas.pydata.org/>`_
* `Xarray <http://xarray.pydata.org/>`_
* `NetCDF4 <https://github.com/Unidata/netcdf4-python>`_
* `Numexpr <https://github.com/pydata/numexpr>`_
* `PyYAML <http://pyyaml.org/>`_
* `Click <http://click.pocoo.org/>`_

`Matplotlib <http://matplotlib.org/>`_ is optional but necessary to graphically display results.

These modules are optional but necessary to display transmission flows on a map:

* NetworkX
* Basemap

These modules are optional and used for the example notebook in the tutorial:

* `Seaborn <https://web.stanford.edu/~mwaskom/software/seaborn/>`_
* `Jupyter <http://jupyter.org/>`_
