.. _installation:

=========================
Download and installation
=========================

Requirements
============

Calliope has been tested on Linux, Mac OS X, and Windows (but see the :ref:`Windows notes <windows_install_note>` below).

Running Calliope requires four things:

1. the Python programming language (version 3)
2. a number of Python add-on modules (see :ref:`below for the complete list <python_module_requirements>`)
3. a solver: Calliope has been tested with `GLPK <https://www.gnu.org/software/glpk/>`_, `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_, and `Gurobi <http://www.gurobi.com/>`_. Any other solver that is compatible with Pyomo, which Calliope uses to construct the models, should work.
4. the Calliope software itself

Installing a solver
===================

Refer to the documentation of your solver on how to install it. You need at least one of the solvers supported by Pyomo installed.

`GLPK <https://www.gnu.org/software/glpk/>`_ is an open-source solver that works very well for smaller models, but can take too much time and/or too much memory on larger problems, for which it can be worth using `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_ or `Gurobi <http://www.gurobi.com/>`_. Both Gurobi and CPLEX have free academic licensing schemes.

Installing Python, required modules and Calliope
================================================

By far the easiest and recommended way to obtain a working Python installation including the required Python modules (items 1 and 2 on the list above) is to use the free `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_.

Once you have Anaconda installed, you can create a new Python 3.5 environment called "calliope" with all the necessary modules with the following command::

   $ conda create -n calliope python=3.5 pip pandas pytables pyyaml matplotlib networkx basemap

Then, with the "calliope" environment activated (``source activate calliope`` if you are using Anaconda), install Calliope with the Python package installer pip, which will also install Pyomo (and any other remaining dependencies not installed already)::

   $ pip install calliope

.. _windows_install_note:

.. Note::

   Calliope has been tested on Windows 7 and Windows 8 and should generally work, but running Python software on Windows can be trickier than on Linux or Mac OS:

   There are some specifics to keep in mind when installing on Windows:

   * On Windows, basemap is not currently available, so plotting maps is unavailable. Use the following command to install the Calliope environment::

      conda create -n calliope python=3.5 pip pandas pytables pyyaml matplotlib

   * To activate an Anaconda environment on Windows, use ``activate`` instead of ``source activate``, e.g: ``activate calliope``

.. _python_module_requirements:

Python module requirements
==========================

If you prefer not to use Anaconda and manually manage your Python setup, these are the modules you need to install.

The following Python modules and their dependencies are required:

* `Pyomo <https://software.sandia.gov/trac/pyomo/wiki/Pyomo>`_
* `Pandas <http://pandas.pydata.org/>`_
* `Numexpr <https://github.com/pydata/numexpr>`_
* `Pytables <https://pytables.github.io/>`_ (which requires `Cython <http://cython.org/>`_ to build from source)
* `PyYAML <http://pyyaml.org/>`_
* `Click <http://click.pocoo.org/>`_

These modules are optional but necessary to display result graphs:

* Matplotlib

These modules are optional but necessary to display transmission flows on a map:

* NetworkX
* Basemap
