.. _installation:

============
Installation
============

Requirements
============

Calliope is only tested on Python 2.7.x and unlikely to work on older versions. Support for 3.x will be implemented as soon as all required modules support Python 3.

The following Python modules are required:

* Coopr (Pyomo)
* NumPy
* Pandas
* PyYAML

In addition,Pyomo requires a solver. Calliope has been tested with both `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_ and `GLPK <https://www.gnu.org/software/glpk/>`_.

The recommended way to obtain all required Python modules with the exception of Coopr/Pyomo is to use the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_. Once that is set up and configured as the active Python interpreter, install Coopr/Pyomo with ``pip install coopr``.

Installation
============

To install the latest stable version via ``pip``::

    pip install -e git+https://github.com/sjpfenninger/calliope.git#egg=calliope

For a more easily modifiable local installation, first clone the repository to a location of your choosing, then install via ``pip``::

   git clone https://github.com/sjpfenninger/calliope
   pip install -e /path/to/your/cloned/repository
