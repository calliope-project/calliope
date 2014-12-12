.. _installation:

=========================
Download and installation
=========================

Requirements
============

Running Calliope requires four things:

1. the Python programming language (version 3.4 or higher)
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

Once you have Anaconda installed, you can create a new Python 3.4 environment called "calliope" with all the necessary modules with the following command::

   $ conda create -n calliope python=3.4 pip pandas pytables pyyaml matplotlib networkx basemap

.. Warning :: Until the release of Pyomo 4.0, Calliope 0.3.0 makes use of the development version of Pyomo. Because of this, for the moment, the following manual installation steps for Pyomo are necessary after setting up the environment, and before installing Calliope itself:

   .. code-block:: bash

      $ pip install svn+https://software.sandia.gov/svn/public/pyutilib/pyutilib/trunk@3457#egg=pyutilib

      $ pip install svn+https://software.sandia.gov/svn/public/pyomo/pyomo/trunk@9471#egg=pyomo

Then, with the "calliope" environment activated (``source activate calliope`` if you are using Anaconda), install Calliope with the Python package installer pip, which will also install Pyomo (and any other remaining dependencies not installed already)::

   $ pip install git+https://github.com/calliope-project/calliope.git#egg=calliope

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
