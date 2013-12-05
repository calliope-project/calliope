
========
Datasets
========

Calliope uses a simple data format to specify those parameters that are explicit in space and time, i.e. those parameters that cannot simply be specified in the model's YAML configuration files.

Structure
=========

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
