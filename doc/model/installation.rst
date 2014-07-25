.. _installation:

============
Installation
============

Requirements
============

Calliope has only been tested on Python 2.7.x and will remain incompatible with Python 3 until Coopr implements Python 3 compatibility.

The following Python modules are required:

* Coopr (Pyomo)
* NumPy
* Pandas
* PyYAML

Optional modules:

* matplotlib (to display results)

In addition, Pyomo requires a solver. Calliope has been tested with both `GLPK <https://www.gnu.org/software/glpk/>`_ and `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_.

The recommended way to obtain all required and optional Python modules with the exception of Coopr is to use the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_. Once that is set up and configured as the active Python interpreter, install Coopr with ``pip install coopr``.

Installation
============

To install the latest stable version via ``pip``::

    pip install -e git+https://github.com/sjpfenninger/calliope.git#egg=calliope

For a more easily modifiable local installation, first clone the repository to a location of your choosing, then install via ``pip``::

   git clone https://github.com/sjpfenninger/calliope
   pip install -e /path/to/your/cloned/repository
