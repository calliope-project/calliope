=================
Development guide
=================

Contributions are very welcome! See our `contributors guide on GitHub <https://github.com/calliope-project/calliope/blob/master/CONTRIBUTING.md>`_ for information on how to contribute.

The code lives on GitHub at `calliope-project/calliope <https://github.com/calliope-project/calliope>`_. Development takes place in the ``master`` branch. Stable versions are tagged off of ``master`` with `semantic versioning <http://semver.org/>`_.

Tests are included and can be run with ``py.test`` from the project's root directory.

Also see the list of `open issues <https://github.com/calliope-project/calliope/issues>`_,  planned `milestones <https://github.com/calliope-project/calliope/milestones>`_ and `projects <https://github.com/calliope-project/calliope/projects>`_ for an overview of where development is heading, and `join us on Gitter <https://gitter.im/calliope-project/calliope>`_ to ask questions or discuss code.

.. _installing_dev:

--------------------------------
Installing a development version
--------------------------------

As when installing a stable version, using ``conda`` is recommended.

To actively contribute to Calliope development, or simply track the latest development version, you'll instead want to clone our GitHub repository. This will provide you with the master branch in a known location on your local device.

First, clone the repository:

  .. code-block:: fishshell

   $ git clone https://github.com/calliope-project/calliope

Then install all development requirements for Calliope into a new environment, calling it e.g. ``calliope_dev``:

  .. code-block:: fishshell

   $ conda env create -f requirements.yml -n calliope_dev
   $ conda activate calliope_dev

Finally install Calliope itself as an editable installation with pip:

  .. code-block:: fishshell

   $ pip install -e calliope

.. Note:: Most of our tests depend on having the CBC solver also installed, as we have found it to be more stable than GPLK. If you are running on a Unix system, then you can run ``conda install coincbc`` to also install the CBC solver. To install solvers other than CBC, and for Windows systems, see our :ref:`solver installation instructions <install_solvers>`.

We use the code formatter `black <https://github.com/psf/black/>`_ and before you contribute any code, you should ensure that you have run it through black. If you don't have a process for doing this already, you can install our configured `pre-commit <https://pre-commit.com/>`_ hook which will automatically run black on each commit:

  .. code-block:: fishshell

   $ pre-commit install

---------------------------
Creating modular extensions
---------------------------

As of version 0.6.0, dynamic loading of custom constraint generator extensions has been removed due it not not being used by users of Calliope. The ability to dynamically load custom functions to adjust time resolution remains (see below).

Time functions and masks
------------------------

Custom functions that adjust time resolution can be loaded dynamically during model initialisation. By default, Calliope first checks whether the name of a function or time mask refers to a function from the :mod:`calliope.core.time.masks` or :mod:`calliope.core.time.funcs` module, and if not, attempts to load the function from an importable module:

.. code-block:: yaml

   time:
      masks:
          - {function: week, options: {day_func: 'extreme', tech: 'wind', how: 'min'}}
          - {function: my_custom_module.my_custom_mask, options: {...}}
      function: my_custom_module.my_custom_function
      function_options: {...}

----------------------------------------------
Understanding Calliope internal implementation
----------------------------------------------

Worried about delving into the Calliope code? Confused by the structure? Fear not! The package is structured as best as possible to follow a clear workflow, which takes inputs on a journey from YAML and CSV files, via Pyomo objects, to a NetCDF file of results.

Overview
-----------------

Calliope enables data stored in YAML and CSV files to be prepared for optimisation in a linear solver, and the results of optimisation to be analysed and/or saved. The internal workflow is shown below. The python packages ruamel.yaml and pandas are used to parse the YAML and CSV files, respectively. Xarray is then used to restructure the data into multidimensional arrays, ready for saving, plotting, or sending to the backend. The pyomo package is currently used in the backend to transform the xarray dataset into a pyomo ConcreteModel. All parameters, sets, constraints, and decision variables are defined as pyomo objects at this stage. Pyomo produces an LP file, which can be read in by the modeller's chosen solver. Results are extracted from pyomo into an xarray dataset, again ready to be analysed or saved.

.. figure:: images/calliope_workflow_basic.*
   :alt: calliope_workflow_overview

Internal implementation
-----------------------

Taking a more detailed look at the workflow, a number of data objects are populated. On initialising a model, the `model_run` dictionary is created from the provided YAML and CSV files. Overrides (both from scenarios and location/link specific ones) are applied at this point. The `model_run` dictionary is then reformulated into multidimensional arrays of data and collated in the `model_data` xarray dataset. At this point, model initialisation has completed; model inputs can be accessed by the user, and edited if necessary.

On executing `model.run()`, only `model_data` is sent over to the backend, where the pyomo `ConcreteModel` is created and pyomo parameters (Param) and sets (Set) are populated using data from `model_data`. Decision variables (Var), constraints (Constraint), and the objective (Obj) are also initialised at this point. The model is then sent to the solver.

Upon solving the problem, the backend_model (pyomo ConcreteModel) is attached to the Model object and the results are added to `model_data`. Post-processing also occurs to clean up the results and to calculate certain indicators, such as the capacity factor of technologies. At this point, the model run has completed; model results can be accessed by the user, and saved or analysed as required.

.. figure:: images/calliope_workflow_complex.*
   :alt: Calliope internal implementation workflow

   Representation of Calliope internal implementation workflow. Five primary steps are shown, starting at the model definition and implemented clockwise. From inner edge to outer edge of the rainbow are: the data object produced by the step, primary and auxiliary python files in which functionality to produce the data object are found, and the folder containing the relevant python files for the step.


Exposing all methods and data attached to the Model object
----------------------------------------------------------

The Model object begins as an empty class. Once called, it becomes an empty object which is populated with methods to access, analyse, and save the model data. The Model object is further augmented once `run` has been called, at which point, the backend model object can be accessed, directly or via a user-friendly interface. The notebook found :nbviewer_docs:`here <_static/notebooks/calliope_model_object.ipynb>` goes through each method and data object which can be accessed through the Model object. Most are hidden (using an underscore before the method name), as they aren't useful for the average user.

.. figure:: images/calliope_model_structure.*
   :alt: Calliope model object augmentation

   Representation of the Calliope Model object, growing from an empty class to having methods to view, plot and save data, and to interface with the solver backend.

---------------------
Contribution workflow
---------------------

Have a bug fix or feature addition you'd like to see in the next stable release of Calliope? First, be sure to check out our list of `open <https://github.com/calliope-project/calliope/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen>`_ and `closed <https://github.com/calliope-project/calliope/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aclosed>`_ issues to see whether this is something someone else has mentioned, or perhaps has even fixed. If it's there, you can add to the discussion, give it a thumbs up, or look to implement the change yourself. If it isn't there, then feel free to open your own issue, or you can head straight to implementing it. The below instructions are a more detailed description of our `contribution guidelines <https://github.com/calliope-project/calliope/blob/master/CONTRIBUTING.md>`_, which you can refer to if you're already comfortable with using pytest and GitHub flows.

Implementing a change
---------------------

When you want to change some part of Calliope, whether it is the software or the documentation, it's best to do it in a fork of the main Calliope project repository. You can find out more about how to fork a repository `on GitHub's help pages <https://help.github.com/articles/fork-a-repo/>`_. Your fork will be a duplicate of the Calliope master branch and can be 'cloned' to provide you with the repository on your own device

  .. code-block:: fishshell

    $ git clone https://github.com/your_username/calliope

If you want the local version of your fork to be in the same folder as your local version of the main Calliope repository, then you just need to specify a new directory name

  .. code-block:: fishshell

    $ git clone https://github.com/your_username/calliope your_new_directory_name

Following the instructions for :ref:`installing a development environment of Calliope <installing_dev>`, you can create an environment specific to this installation of Calliope.

In making changes to your local version, it's a good idea to create a branch first, to not have your master branch diverge from that of the main Calliope repository

  .. code-block:: fishshell

    $ git branch new-fix-or-feature

Then, 'checkout' the branch so that the folder contents are specific to that branch

  .. code-block:: fishshell

    $ git checkout new-fix-or-feature

Finally, push the branch online, so it's existence is also in your remote fork of the Calliope repository (you'll find it in the dropdown list of branches at https://github.com/your_repository/calliope)

  .. code-block:: fishshell

    $ git push -u origin new-fix-or-feature

Now the files in your local directory can be edited with complete freedom. Once you have made the necessary changes, you'll need to test that they don't break anything. This can be done easily by changing to the directory into which you cloned your fork using the terminal / command line, and running `pytest <https://docs.pytest.org/en/latest/index.html>`_ (make sure you have activated the conda environment and you have pytest installed: `conda install pytest`). Any change you make should also be covered by a test. Add it into the relevant test file, making sure the function starts with 'test\_'. Since the whole test suite takes ~25 minutes to run, you can run specific tests, such as those you add in

  .. code-block:: fishshell

    $ pytest calliope/test/test_filename.py::ClassName::function_name

If tests are failing, you can debug them by using the pytest arguments ``-x`` (stop at the first failed test) and ``--pdb`` (enter into the debug console).

Once everything has been updated as you'd like (see the contribution checklist below for more on this), you can commit those changes. This stores all edited files in the directory, ready for pushing online

  .. code-block:: fishshell

    $ git add .
    $ git checkout -m "Short message explaining what has been done in this commit."

If you only want a subset of edited files to go into this commit, you can specify them in the call to `git add`; the period adds all edited files.

If you're happy with your commit(s) then it is time to 'push' everything online using the command `git push`. If you're working with someone else on a branch and they have made changes, you can bring them into your local repository using the command `git pull`.

Now it is time to request that these changes are added into the main Calliope project repository! You can do this by starting a `pull request <https://help.github.com/articles/about-pull-requests/>`_. One of the core Calliope team will review the pull request and either accept it or request some changes before it's merged into the main Calliope repository. If any changes are requested, you can make those changes on your local branch, commit them, and push them online -- your pull request will update automatically with those changes.

Once a pull request has been accepted, you can return your fork back to its master branch and `sync it <https://help.github.com/articles/syncing-a-fork/>`_ with the updated Calliope project master

  .. code-block:: fishshell

   $ git remote add upstream https://github.com/calliope-project/calliope
   $ git fetch upstream master
   $ git checkout master
   $ git merge upstream/master

Contribution checklist
----------------------

A contribution to the core Calliope code should meet the following requirements:

   1. Test(s) added to cover contribution

      Tests ensure that a bug you've fixed will be caught in future, if an update to the code causes it to occur again. They also allow you to ensure that additional functionality works as you expect, and any change elsewhere in the code that causes it to act differently in future will be caught.

   2. Documentation updated

      If you've added functionality, it should be mentioned in the documentation. You can find the reStructuredText (.rst) files for the documentation under 'doc/user'.

   3. Changelog updated

      A brief description of the bug fixed or feature added should be placed in the changelog (changelog.rst). Depending on what the pull request introduces, the description should be prepended with `fixed`, `changed`, or `new`.

   4. Coverage maintained or improved

      Coverage will be shown once all tests are complete online. It is the percentage of lines covered by at least one test. If you've added a test or two, you should be fine. But if coverage does go down it means that not all of your contribution has been tested!

   .. figure:: images/coveralls.*
      :alt: Example of coverage notification on a pull request

      Example of coverage notification in a pull request.

If you're not sure you've done everything to have a fully formed pull request, feel free to start it anyway. We can help guide you through making the necessary changes, once we have seen where you've got to.

---------
Profiling
---------

To profile a Calliope run with the built-in national-scale example model, then visualise the results with snakeviz:

.. code-block:: shell

   make profile  # will dump profile output in the current directory
   snakeviz calliope.profile  # launch snakeviz to visually examine profile


Use ``mprof plot`` to plot memory use.

Other options for visualising:

* Interactive visualisation with `KCachegrind <https://kcachegrind.github.io/>`_ (on macOS, use QCachegrind, installed e.g. with ``brew install qcachegrind``)

   .. code-block:: shell

      pyprof2calltree -i calliope.profile -o calliope.calltree
      kcachegrind calliope.calltree

* Generate a call graph from the call tree via graphviz

   .. code-block:: shell

      # brew install gprof2dot
      gprof2dot -f callgrind calliope.calltree | dot -Tsvg -o callgraph.svg

-------------------------
Checklist for new release
-------------------------

Pre-release
-----------

* Make sure all unit tests pass
* Build up-to-date Plotly plots for the documentation with (``make doc-plots``)
* Re-run tutorial Jupyter notebooks, found in `doc/_static/notebooks`
* Make sure documentation builds without errors
* Make sure the release notes are up-to-date, especially that new features and backward incompatible changes are clearly marked

Create release
--------------

* Change ``_version.py`` version number
* Update changelog with final version number and release date
* Commit with message "Release vXXXX", then add a "vXXXX" tag, push both to GitHub
* Create a release through the GitHub web interface, using the same tag, titling it "Release vXXXX" (required for Zenodo to pull it in)
* Upload new release to PyPI: ``make all-dist``
* Update the conda-forge package:
    * Fork `conda-forge/calliope-feedstock <https://github.com/conda-forge/calliope-feedstock>`_, and update ``recipe/meta.yaml`` with:
        * Version number: ``{% set version = "XXXX" %}``
        * SHA256 of latest version from PyPI: ``{% set sha256 = "XXXX" %}``
        * Reset ``build: number: 0`` if it is not already at zero
        * If necessary, carry over any changed requirements from ``setup.py`` or ``requirements/base.yml``
    * Submit a pull request from an appropriately named branch in your fork (e.g. ``vXXXX``) to the `conda-forge/calliope-feedstock <https://github.com/conda-forge/calliope-feedstock>`_ repository

Post-release
------------

* Update changelog, adding a new vXXXX-dev heading, and update ``_version.py`` accordingly, in preparation for the next master commit

* Update the ``calliope_version`` setting in all example models to match the new version, but without the ``-dev`` string (so ``0.6.0-dev`` is ``0.6.0`` for the example models)

.. Note:: Adding '-dev' to the version string, such as ``__version__ = '0.1.0-dev'``, is required for the custom code in ``doc/conf.py`` to work when building in-development versions of the documentation.
