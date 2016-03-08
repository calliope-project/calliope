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


Installing Python, required modules and Calliope
================================================

By far the easiest and recommended way to obtain a working Python installation including the required Python modules (items 1 and 2 on the list above) is to use the free `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_. On systems with limited disk space, use the `Miniconda distribution <http://conda.pydata.org/miniconda.html>`_, which does not come with any pre-included packages, downloading packages only as required.

Once you have Anaconda/Miniconda installed, you can create a new Python 3.5 environment called "calliope" with all the necessary modules with the following command (see the note on Windows below if this command causes an error)::

   $ conda create -n calliope python=3.5 pip pandas pytables pyyaml matplotlib networkx basemap seaborn jupyter notebook --yes

Then, you need to activate the "calliope" environment. On Linux and Mac OS X::

   $ source activate calliope

On Windows::

   $ activate calliope

Finally, install Calliope with the Python package installer pip, which will also automatically install Pyomo (and any other remaining dependencies not installed already)::

   $ pip install calliope

.. _windows_install_note:

.. Note::

   Calliope has been tested on Windows 7 and Windows 8 and should generally work, but running Python software on Windows can be trickier than on Linux or Mac OS:

   Note that on Windows, basemap for Python 3.x is not currently available for Anaconda, so you need to `manually install it <http://matplotlib.org/basemap/users/installing.html>`_ if you want to plot maps. Use the following command, which does not include basemap, to install the Calliope environment on Windows::

      $ conda create -n calliope python=3.5 pip pandas pytables pyyaml matplotlib networkx seaborn jupyter notebook --yes


Installing a solver
===================

You need at least one of the solvers supported by Pyomo installed. GLPK or Gurobi are recommended and have been confirmed to work with Calliope. Refer to the documentation of your solver on how to install it. Some details on GLPK and Gurobi are given below. Another commercial alternative is `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_.

GLPK
----

`GLPK <https://www.gnu.org/software/glpk/>`_ is free and open-source, but can take too much time and/or too much memory on larger problems.

On Windows, it can be easily installed into the "calliope" environment (make sure the environment has been activated as shown above)::

   $ conda install -c sjpfenninger glpk


For Linux and Mac OS X, refer to the `GLPK website <https://www.gnu.org/software/glpk/>`_ for installation instructions.

Gurobi
------

`Gurobi <http://www.gurobi.com/>`_ is commercial but significantly faster than GLPK, which is relevant for larger problems. It needs a license to work, which can be obtained for free for academic use by making an account on gurobi.com.

On Windows and Linux, Gurobi can be installed via conda::

    $ conda install -c gurobi gurobi

On Mac OS X, it has to be downloaded manually from the `Gurobi website <http://www.gurobi.com/>`_.

After installing, log on to gurobi.com and obtain a (free or paid) license, then activate it on your system via the instructions given online (using the ``grbgetkey`` command).

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

These modules are optional but necessary to graphically display results:

* Matplotlib

These modules are optional but necessary to display transmission flows on a map:

* NetworkX
* Basemap

These modules are optional and used for the example notebook in the tutorial:

* `Seaborn <https://web.stanford.edu/~mwaskom/software/seaborn/>`_
* `Jupyter <http://jupyter.org/>`_
