# Contributing

Calliope is an actively maintained and utilised project.

## How to contribute

to report issues, request features, or exchange with our community, just follow the links below.

__Is something not working?__

[:material-bug: Report a bug](https://github.com/calliope-project/calliope/issues/new?template=BUG-REPORT.yml "Report a bug in calliope by creating an issue and a reproduction"){ .md-button }

__Missing information in our docs?__

[:material-file-document: Report a docs issue](https://github.com/calliope-project/calliope/issues/new?template=DOCS.yml "Report missing information or potential inconsistencies in our documentation"){ .md-button }

__Want to submit an idea?__

[:material-lightbulb-on: Request a change](https://github.com/calliope-project/calliope/issues/new?template=FEATURE-REQUEST.yml "Propose a change or feature request or suggest an improvement"){ .md-button }

__Have a question or need help?__

[:material-chat-question: Ask a question](https://github.com/calliope-project/calliope/discussions "Ask questions on our discussion board and get in touch with our community"){ .md-button }

## Developing calliope

To find beginner-friendly existing bugs and feature requests you may like to start out with, take a look at our [good first issues](https://github.com/calliope-project/calliope/contribute).

### Setting up a development environment

To create a development environment for calliope, with all libraries required for development and quality assurance installed, it is easiest to install calliope using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager, as follows:

1. Install mamba with the [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) executable for your operating system.
1. Open the command line (or the "miniforge prompt" in Windows).
1. Download (a.k.a., clone) the calliope repository: `git clone git@github.com:calliope-project/calliope.git`
1. Change into the `calliope` directory: `cd calliope`
1. Create the calliope mamba environment: `mamba create -n calliope -c conda-forge --file requirements/base.txt --file requirements/dev.txt`
1. Activate the calliope mamba environment: `mamba activate calliope`
1. Install the calliope package into the environment, in editable mode and ignoring dependencies (we have dealt with those when creating the mamba environment): `pip install --no-deps -e .`

All together:

``` shell
git clone git@github.com:calliope-project/calliope.git
cd calliope
mamba create -n calliope -c conda-forge --file requirements/base.txt
mamba activate calliope
pip install --no-deps -e .
```

If installing directly with pip, you can install these libraries using the `dev` option, i.e., `pip install -e '.[dev]'`

If you plan to make changes to the code then please make regular use of the following tools to verify the codebase while you work:

- `pre-commit`: run `pre-commit install` in your command line to load inbuilt checks that will run every time you commit your changes.
The checks are: 1. check no large files have been staged, 2. lint python files for major errors, 3. format python files to conform with the [PEP8 standard](https://peps.python.org/pep-0008/).
You can also run these checks yourself at any time to ensure staged changes are clean by calling `pre-commit`.
- `pytest` - run the unit test suite and check test coverage.

!!! note
    If you already have an environment called `calliope` on your system (e.g., for a stable installation of the package), you will need to choose a different environment name, e.g. `calliope-dev`.

### Implementing a change

When you want to change some part of Calliope, whether it is the software or the documentation, it's best to do it in a fork of the main Calliope project repository.
You can find out more about how to fork a repository on [GitHub's help pages](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
Your fork will be a duplicate of the Calliope main branch and can be 'cloned' to provide you with the repository on your own device.

```fishshell
git clone https://github.com/your_username/calliope
```

If you want the local version of your fork to be in the same folder as your local version of the main Calliope repository, then you just need to specify a new directory name:

```fishshell
git clone https://github.com/your_username/calliope your_new_directory_name
```

Following the instructions for [installing a development environment of Calliope][setting-up-a-development-environment], you can create an environment specific to this installation of Calliope.

In making changes to your local version, it's a good idea to create a branch first, to not have your main branch diverge from that of the main Calliope repository:

```fishshell
git branch new-fix-or-feature
```

Then, 'checkout' the branch so that the folder contents are specific to that branch:

```fishshell
git checkout new-fix-or-feature
```

Finally, push the branch online, so it's existence is also in your remote fork of the Calliope repository (you'll find it in the dropdown list of branches at https://github.com/your_repository/calliope):

```fishshell
git push -u origin new-fix-or-feature
```

Now the files in your local directory can be edited with complete freedom.
Once you have made the necessary changes, you'll need to test that they don't break anything.
This can be done easily by changing to the directory into which you cloned your fork using the terminal / command line, and running [pytest](https://docs.pytest.org/en/latest/index.html)
Any change you make should also be covered by a test.
Add it into the relevant test file, making sure the function starts with 'test\_'.

If tests are failing, you can debug them by using the pytest arguments ``-x`` (stop at the first failed test) and ``--pdb`` (enter into the debug console).

### Rapid-fire testing

The following options allow you to strip down the test suite to the bare essentials:

1. The test suite includes unit tests and integration tests.
The integration tests can be slow, so if you want to avoid them during development, you should run `pytest -m "not time_intensive"` to ignore those tests flagged as `time_intensive`.
1. You can avoid tracking code coverage (which can be slow), by adding the `--no-cov` argument: `pytest --no-cov`.

All together:

``` shell
pytest -m "not time_intensive" --no-cov
```

If you are developing your own tests, you can focus on those with the `::` syntax:

``` shell
pytest tests/test_my_tests.py::TestMyTestClass::test_my_test_function
```

### Committing changes

Once everything has been updated as you'd like (see the contribution checklist below for more on this), you can commit those changes.
This stores all edited files in the directory, ready for pushing online

```fishshell
git add .
git checkout -m "Short message explaining what has been done in this commit."
```

If you only want a subset of edited files to go into this commit, you can specify them in the call to `git add`; the period adds all edited files.

If you're happy with your commit(s) then it is time to 'push' everything online using the command `git push`.
If you're working with someone else on a branch and they have made changes, you can bring them into your local repository using the command `git pull`.

Now it is time to request that these changes are added into the main Calliope project repository!
You can do this by starting a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).
One of the core Calliope team will review the pull request and either accept it or request some changes before it's merged into the main Calliope repository.
If any changes are requested, you can make those changes on your local branch, commit them, and push them online -- your pull request will update automatically with those changes.

Once a pull request has been accepted, you can return your fork back to its main branch and [sync it](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) with the updated Calliope project `main` branch:

```fishshell
git remote add upstream https://github.com/calliope-project/calliope
git fetch upstream main
git checkout main
git merge upstream/main
```

## Submitting changes

--8<-- "CONTRIBUTING.md:docs"
