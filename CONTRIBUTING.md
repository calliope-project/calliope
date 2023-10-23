# How to contribute

We're really glad you're reading this, because we need volunteer developers to help this project come to fruition.

Some of the resources to look at if you're interested in contributing:

* [Join us on Gitter to chat!](https://app.gitter.im/#/room/#calliope-project_calliope:gitter.im)
* Look at our [milestones](https://github.com/calliope-project/calliope/milestones) and [projects](https://github.com/calliope-project/calliope/projects) on GitHub for an idea on where development is headed
* Look at [open issues tagged with "help wanted"](https://github.com/calliope-project/calliope/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) and ["good first issue"](https://github.com/calliope-project/calliope/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
* Look at the [development guide in our documentation](http://calliope.readthedocs.io/en/latest/user/develop.html)

## Licensing

By contributing to Calliope, e.g. through opening a pull request or submitting a patch, you represent that your contributions are your own original work and that you have the right to license them, and you agree that your contributions are licensed under the Apache 2.0 license.

## Submitting bug reports

[Open an issue on GitHub](https://github.com/calliope-project/calliope/issues/new) to report bugs or other problems.

If reporting an error when running Calliope on the command line, please re-run your command with the ``--debug`` option, e.g.:

``calliope run my_model.yaml --debug``

Then post the full output from the debug run as part of your GitHub issues.

If reporting an error when running Calliope interactively in a Python session, please include a full traceback in your issue.

## Submitting changes

Look at the [development guide in our documentation](http://calliope.readthedocs.io/en/latest/user/develop.html) for information on how to get set up for development.

To contribute changes:

1. Fork the project on GitHub
2. Create a feature branch to work on in your fork (``git checkout -b new-fix-or-feature``)
3. Add your name to the ``AUTHORS`` file
4. Commit your changes to the feature branch after running black to format your code (formatting is automatic if the ``pre-commit`` hooks have been installed; see [below](#codeconvention) for more info)
5. Push the branch to GitHub (``git push origin new-fix-or-feature``)
6. On GitHub, create a new [pull request](https://github.com/calliope-project/calliope/pull/new/main) from the feature branch

Our [development guide](http://calliope.readthedocs.io/en/latest/user/develop.html) gives a more detailed description of each step, if you're new to working with GitHub.

### Pull requests

Before submitting a pull request, check whether you have:

* Added your changes to ``changelog.rst``
* Added or updated documentation for your changes
* Added tests if you implemented new functionality

When opening a pull request, please provide a clear summary of your changes!

### Commit messages

Please try to write clear commit messages. One-line messages are fine for small changes, but bigger changes should look like this:

    A brief summary of the commit

    A paragraph or bullet-point list describing what changed and its impact,
    covering as many lines as needed.

## Testing

We have existing test coverage for the key functionality of Calliope.

All tests are in the ``tests`` directory and use [pytest](https://docs.pytest.org/en/latest/).

Our test coverage is not perfect and an easy way to contribute code is to work on better tests.

## <a name="codeconvention"></a>Coding conventions

Start reading our code and you'll get the hang of it.

We mostly follow the official [Style Guide for Python Code (PEP8)](https://www.python.org/dev/peps/pep-0008/).

We have chosen to use the uncompromising code formatter, [`black`](https://github.com/psf/black/).  If run from the root directory of this repo, `pyproject.toml` should ensure the line lengths are restricted to 88.  The philosophy behind using black is to have uniform style throughout the project dictated by code.  Since `black` is designed to minimise diffs, and make patches more human readable, this also makes code reviews more efficient.  To make this a smooth experience, you should add a black formatting script to your git pre-commit hooks before creating a PR.  We provide a quick and easy way to set this up as part of the [development guide in our documentation](http://calliope.readthedocs.io/en/latest/user/develop.html#installing-a-development-version).

## Attribution

The layout and content of this document is partially based on the [OpenGovernment project's contribution guidelines](https://github.com/opengovernment/opengovernment/blob/master/CONTRIBUTING.md).
