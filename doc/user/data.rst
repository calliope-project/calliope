
========
Datasets
========

Calliope uses a simple data format to specify those parameters that are explicit in space and time, i.e. those parameters that cannot simply be specified in the model's YAML configuration files.

---------
Structure
---------

A dataset is simply a folder with CSV files laid out as follows:

.. code-block:: text

   dataset/
      set_t.csv
      tech_r.csv
      tech_r_eff.csv
      tech_e_eff.csv
      ...

The only mandatory file is ``set_t.csv``, which defines the possible time steps (at the maximum possible resolution) for the given model. It must contain two columns: the first one, integer indices, the second, date-time strings formatted as ``YYYY-MM-DD hh:mm:ss``, e.g. ``2005-01-01 00:00:00``.

The rest of the files are named according to the scheme ``{tech}_{parameter}.csv``, where ``{tech}`` is the name of the technology and ``{parameter}`` is the parameter defined by this CSV file, for example ``wind_r.csv`` or ``pv_r_eff.csv``.

Each CSV file must have integer indices in the first column, column names in the first row, and integer or floating point values in the remaining cells, for example:

.. code-block:: text

   ,node0,node1,node2,...
   0,10,20,10.0,...
   1,11,19,9.9,...
   2,12,18,9.8,...
   ...

The integer indices in the CSV file's first column must match the first column of ``set_t.csv``.

The column names must either match with node names defined in ``nodes.yaml`` or, a given node may select an arbitrary column by defining an ``x_map: 'node_in_csv: node_in_nodes_yaml'`` setting for a given technology, for example:

.. code-block:: yaml

   nodes:
      r1:
         level: 1
         within:
         techs ['demand']
         override:
            demand:
               x_map: 'node_in_csv: r1'

----------------------------------
Managing time steps and resolution
----------------------------------

The default time step length is 1 hour. However, this 1-hourly resolution can be adjusted over parts of the dataset by using the :class:`~calliope.TimeSummarizer` class (currently, only support for downsampling is implemented).

There are two ways to adjust resolution:

1. The :meth:`calliope.TimeSummarizer.reduce_resolution` method: reduces resolution over the entire range of data to the given resolution.
2. The :meth:`calliope.TimeSummarizer.dynamic_timestepper` method: reduces resolution dynamically according to a given mask, allowing to keep high resolution in areas of interest while reducing computational complexity elsewhere.

Dynamic timesteps and masks
---------------------------

In order to use ``dynamic_timestepper``, a mask needs to be generated first.

A mask is a pandas DataFrame with the same index as the data it applies to, and a column called ``summarize`` (in addition to any number of additional columns, which are ignored). The ``summarize`` column containts ``0`` for timesteps that aren't touched, and blocks starting with an integer >1 and followed by the integer's value number of ``-1``, for timesteps that are to be summarized. For example::

   [0, 0, 0, 3, -1, -1, 3, -1, -1, 0, 0, 0]

The above example means "compress the 4th-6th and 7th-9th timesteps into two new timesteps with a resolution of 3".

Functions to generate masks and resolution series are in ``calliope.time_functions``.

.. FIXME this needs updating

A fully-functioning example of using a mask to collapse periods where solar irradiance is zero (i.e., the night) into single timesteps::

   model = calliope.Model()
   s = calliope.TimeSummarizer()
   mask = calliope.time_functions.mask_where_zero(model.data, tech='csp', var='r')
   s.dynamic_timestepper(model.data, mask)  # Modifies data in place
   model.run()
