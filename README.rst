Calliope
========

*A multi-scale energy systems (MUSES) modeling framework*

Work in progress.

Requirements
------------

Calliope is only tested on Python 2.7.x.

Required modules:

* Coopr (Pyomo)
* NumPy
* Pandas
* PyYAML

Optional modules:

* matplotlib

In addition,Pyomo requires a solver. Calliope has been tested with both `GLPK <https://www.gnu.org/software/glpk/>`_ and `CPLEX <http://ibm.com/software/integration/optimization/cplex-optimization-studio/>`_.

The recommended way to obtain all required Python modules with the exception of Coopr/Pyomo is to use the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_. Once that is set up and configured as the active Python interpreter, install Coopr/Pyomo with ``pip install coopr``.

Installation
------------

To install the latest stable version::

   pip install -e git+https://github.com/sjpfenninger/calliope.git#egg=calliope

Documentation
-------------

Documentation is available at `calliope.readthedocs.org <https://calliope.readthedocs.org/>`_.

License
-------

Copyright 2013 Stefan Pfenninger and released under an `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ license.
