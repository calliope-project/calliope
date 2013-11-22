
==============
Implementation
==============

Overriding options:

1. Subclassing a technology, e.g. defining an override to read a different file:

.. code-block:: yaml

   demand-eu:
      parent: 'demand'
      constraints:
         r_scale_to_peak: -60000

2. Manually specifying file name to read:

.. code-block:: yaml

   demand:
      constraints:
         r: 'file=demand-eu_r.csv'
         r_scale_to_peak: -60000

-----------------------
How parameters are read
-----------------------

If a parameter is not explicit in time and space, it is simply read from ``model_settings.yaml`` as needed during model generation, using the ``get_option()`` method.

If a parameter is explicit in time and space, it is read and stored in the :class:`~calliope.Model` object's ``data`` attribute during its instantiation (in ``read_data()``).

There are various limitations in how this happens, which make some combinations of custom values difficult. However, it is possible to modify them manually after instantiation and before calling ``generate_model()``.

The parameters this currently applies to are:

* ``r``
* ``r_eff``
* ``e_eff``

The steps taken for each of these parameters ``param``, for technology ``y``, are:

1. Check if the parameter is defined in ``model_settings.yaml`` for ``y``. If so, the value is read and stored (in ``read_data()``) and later set as the parameter value for all ``x, t`` (in ``generate_model()``).

2. If (1) does not apply, try loading the parameter from a CSV file, with the format ``{y}_{param}.csv``, so for example ``pv_r.csv`` for a PV resource parameter. The CSV file must contain timesteps as rows and nodes as columns.

.. admonition:: Note

   After reading the CSV file, if any columns are missing (i.e. if a file does not contain columns for all nodes in the current :class:`~calliope.Model`'s nodes set), they are added with a value of 0 for all timesteps.


3. If neither (1) nor (2) apply, the value is read analogously to (1) but from the ``default`` technology in ``model_settings.yaml``.

``read_data()`` will fail if neither of these three steps work for a parameter.