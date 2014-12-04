
=================
Development guide
=================

.. TODO intro text

--------------------------------
Installing a development version
--------------------------------

For a more easily modifiable local installation, first clone the repository to a location of your choosing, then install via ``pip``::

   git clone https://github.com/sjpfenninger/calliope
   pip install -e /path/to/your/cloned/repository

-------------------------
Adding custom constraints
-------------------------

.. TODO

-----------------------------------------
Modifying a Model instance before solving
-----------------------------------------

Parameters read from CSV files are read and stored in the :class:`~calliope.Model` object's ``data`` attribute during its instantiation (in ``read_data()``).

There are various limitations in how this happens, which make some combinations of custom values difficult. However, it is always possible to modify the data manually after model instantiation before calling ``generate_model()``.
