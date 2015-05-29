
This is a non-exhaustive overview of planned improvements. There is no fixed time plan. Contributions are welcome, see the `development guide <http://docs.callio.pe/en/latest/user/develop.html>`_.

Upcoming release plans
======================

v0.4.0
------

* Mixed integer (MILP) constraints
* Dynamic conditional constraints: resource for ``supply`` techs can temporarily become negative to emulate parasitics or sub-sections of a connected grid

v0.5.0
------

* Better time mask functions
* Multi-run configuration generator, e.g. with the ability to draw values for a specific parameter from a given distribution
* Configuration aliases
* Metadata handling in configuration objects
    - Keep track of source file and line of each setting
    - Permit dynamically updating configuration files
