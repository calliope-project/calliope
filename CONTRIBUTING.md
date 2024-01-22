# How to contribute

We're really glad you're reading this, because we need volunteer developers to help this project come to fruition.

Some of the resources to look at if you're interested in contributing:

* [Join us on Gitter to chat!](https://app.gitter.im/#/room/#calliope-project_calliope:gitter.im).
* [Join or start a discussion thread on GitHub](https://github.com/calliope-project/calliope/discussions).
* Look at our [milestones](https://github.com/calliope-project/calliope/milestones) and [projects](https://github.com/calliope-project/calliope/projects) on GitHub for an idea on where development is headed.
* Look at [open issues tagged with "help wanted"](https://github.com/calliope-project/calliope/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) and ["good first issue"](https://github.com/calliope-project/calliope/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
* Look at the [development guide in our documentation](http://calliope.readthedocs.io/en/stable/contributing)

## Licensing

By contributing to Calliope, e.g. through opening a pull request or submitting a patch, you represent that your contributions are your own original work and that you have the right to license them, and you agree that your contributions are licensed under the Apache 2.0 license.

## Submitting bug reports

You can open an issue on GitHub to report bugs or request new Calliope features.
Follow these links to submit your issue:

* [Report bugs or other problems while running calliope](https://github.com/calliope-project/calliope/issues/new?template=BUG-REPORT.yml).
If reporting an error, please include a full traceback in your issue.
* [Request features that calliope does not already include](https://github.com/calliope-project/calliope/issues/new?template=FEATURE-REQUEST.yml).
* [Report missing or inconsistent information in our documentation](https://github.com/calliope-project/calliope/issues/new?template=DOCS.yml).
* [Any other issue](https://github.com/calliope-project/calliope/issues/new).

If reporting an error when running Calliope on the command line, please re-run your command with the `--debug` option, e.g.:

```shell
calliope run my_model.yaml --debug
```

Then post the full output from the debug run as part of your GitHub issues.

If reporting an error when running Calliope interactively in a Python session, please include a full traceback in your issue.

## Submitting changes

Look at the [development guide in our documentation](http://calliope.readthedocs.io/en/stable/contributing) for information on how to get set up for development.

To contribute changes:

1. Fork the project on GitHub
2. Create a feature branch to work on in your fork (`git checkout -b new-fix-or-feature`)
3. Add your name to the `AUTHORS` file
4. Commit your changes to the feature branch after running black to format your code (formatting is automatic if the `pre-commit` hooks have been installed; see [below](#code-conventions) for more info)
5. Push the branch to GitHub (`git push origin new-fix-or-feature`)
6. On GitHub, create a new [pull request](https://github.com/calliope-project/calliope/pull/new/main) from the feature branch

<!--- the "--8<--" html comments define what part of this file to add to the index page of the documentation -->
<!--- --8<-- [start:docs] -->
### Pull requests

Before submitting a pull request, check whether you have:

1. **Test(s) added to cover contribution**.
Tests ensure that a bug you've fixed will be caught in future, if an update to the code causes it to occur again.
They also allow you to ensure that additional functionality works as you expect, and any change elsewhere in the code that causes it to act differently in future will be caught.
2. **Updated the documentation**.
If you've added functionality, it should be mentioned in the documentation. You can find the Markdown files for the documentation in the 'docs' directory.
3. **Updated the changelog**.
A brief description of the bug fixed or feature added should be placed in the changelog (CHANGELOG.md).
Depending on what the pull request introduces, the description should be prepended with `fixed`, `changed`, `added` or `new`.
4. **maintained or improved code coverage**.
Coverage will be shown once all tests are complete online.
It is the percentage of lines covered by at least one test.
If you've added a test or two, you should be fine.
But if coverage does go down it means that not all of your contribution has been tested!

If you're not sure you've done everything to have a fully formed pull request, feel free to start it anyway.
We can help guide you through making the necessary changes, once we have seen where you've got to.

### Commit messages

Please try to write clear commit messages.
One-line messages are fine for small changes, but bigger changes should look like this:

```plain
A brief summary of the commit

A paragraph or bullet-point list describing what changed and its impact,
covering as many lines as needed.
```

### Code conventions

Start reading our code and you'll get the hang of it.

We mostly follow the official [Style Guide for Python Code (PEP8)](https://www.python.org/dev/peps/pep-0008/).

We have chosen to use the uncompromising code formatter [`black`](https://github.com/psf/black/) and the linter [`ruff`](https://beta.ruff.rs/docs/).
When run from the root directory of this repository, `pyproject.toml` should ensure that formatting and linting fixes are in line with our custom preferences (e.g., 88 character maximum line length).
The philosophy behind using `black` is to have uniform style throughout the project dictated by code.
Since `black` is designed to minimise diffs, and make patches more human readable, this also makes code reviews more efficient.
To make this a smooth experience, you should run `pre-commit install` after setting up your development environment, so that `black` makes all the necessary fixes to your code each time you commit, and so that `ruff` will highlight any errors in your code.
If you prefer, you can also set up your IDE to run these two tools whenever you save your files, and to have `ruff` highlight erroneous code directly as you type.
Take a look at their documentation for more information on configuring this.

We require all new contributions to have docstrings for all modules, classes and methods.
When adding docstrings, we request you use the [Google docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Release checklist
<!--- TODO -->

<!--- --8<-- [end:docs] -->

## Attribution

The layout and content of this document is partially based on the [OpenGovernment project's contribution guidelines](https://github.com/opengovernment/opengovernment/blob/master/CONTRIBUTING.md).
