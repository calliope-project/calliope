[![Chat on Gitter](https://img.shields.io/gitter/room/calliope-project/calliope.svg?style=flat-square)](https://gitter.im/calliope-project/calliope)
[![Master branch build status](https://img.shields.io/azure-devops/build/calliope-project/371cbbaa-fa6b-4efb-9b23-c4283a8e33eb/1?style=flat-square)](https://dev.azure.com/calliope-project/calliope/_build?definitionId=1)
[![Documentation build status](https://img.shields.io/readthedocs/calliope.svg?style=flat-square)](https://readthedocs.org/projects/calliope/builds/)
[![Test coverage](https://img.shields.io/codecov/c/github/calliope-project/calliope?style=flat-square&token=b4fd170f0e7b43679a8bf649719e1cea)](https://codecov.io/gh/calliope-project/calliope)
[![PyPI version](https://img.shields.io/pypi/v/calliope.svg?style=flat-square)](https://pypi.python.org/pypi/calliope)
[![Anaconda.org/conda-forge version](https://img.shields.io/conda/vn/conda-forge/calliope.svg?style=flat-square&label=conda)](https://anaconda.org/conda-forge/calliope)
[![JOSS DOI](https://img.shields.io/badge/JOSS-10.21105/joss.00825-green.svg?style=flat-square)](https://doi.org/10.21105/joss.00825)

---

<img src="https://raw.githubusercontent.com/calliope-project/calliope/master/doc/_static/logo.png" width="364">

*A multi-scale energy systems modelling framework* | [www.callio.pe](http://www.callio.pe/)

---

## Contents

- [Contents](#contents)
- [About](#about)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [What's new](#whats-new)
- [Citing Calliope](#citing-calliope)
- [License](#license)

---

## About

Calliope is a framework to develop energy system models, with a focus on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data). Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents. In an optional operational it can also test a pre-defined system under different operational conditions.

A Calliope model consists of a collection of text files (in YAML and CSV formats) that fully define a model, with details on technologies, locations, resource potentials, etc. Calliope takes these files, constructs an optimization problem, solves it, and reports back results. Results can be saved to CSV or NetCDF files for further processing, or analysed directly in Python through Python's extensive scientific data processing capabilities provided by libraries like [Pandas](http://pandas.pydata.org/) and [xarray](http://xarray.pydata.org/).

Calliope comes with several built-in analysis and visualisation tools. Having some knowledge of the Python programming language helps when running Calliope and using these tools, but is not a prerequisite.

## Quick start

Calliope can run on Windows, macOS and Linux. Installing it is quickest with the `conda` package manager by running a single command: `conda create -c conda-forge -n calliope calliope`.

See the documentation for more [information on installing](https://calliope.readthedocs.io/en/stable/user/installation.html), including what to do if you are having issues with `conda`.

Several easy to understand example models are [included with Calliope](calliope/example_models) and accessible through the `calliope.examples` submodule.

The [tutorials in the documentation run through these examples](https://calliope.readthedocs.io/en/stable/user/tutorials.html). A good place to start is to look at these tutorials to get a feel for how Calliope works, and then to read the "Introduction", "Building a model", "Running a model", and "Analysing a model" sections in the online documentation.

More fully-featured examples that have been used in peer-reviewed scientific publications are available in our [model gallery](https://www.callio.pe/model-gallery/).

## Documentation

Documentation is available on Read the Docs:

* [Read the documentation online (recommended)](https://calliope.readthedocs.io/en/stable/)
* [Download all documentation in a single PDF file](https://readthedocs.org/projects/calliope/downloads/pdf/stable/)

## Contributing

To contribute changes:

1. Fork the project on GitHub
2. Create a feature branch to work on in your fork (`git checkout -b new-feature`)
3. Add your name to the AUTHORS file
4. Commit your changes to the feature branch
5. Push the branch to GitHub (`git push origin my-new-feature`)
6. On GitHub, create a new pull request from the feature branch

See our [contribution guidelines](https://github.com/calliope-project/calliope/blob/master/CONTRIBUTING.md) for more information -- and [join us on Gitter](https://gitter.im/calliope-project/calliope) to ask questions or discuss code.

## What's new

See changes made in recent versions in the [changelog](https://github.com/calliope-project/calliope/blob/master/changelog.rst).

## Citing Calliope

If you use Calliope for academic work please cite:

Stefan Pfenninger and Bryn Pickering (2018). Calliope: a multi-scale energy systems modelling framework. *Journal of Open Source Software*, 3(29), 825. [doi: 10.21105/joss.00825](https://doi.org/10.21105/joss.00825)

## License

Copyright since 2013 Calliope contributors listed in AUTHORS

Licensed under the Apache License, Version 2.0 (the "License"); you
may not use this file except in compliance with the License. You may
obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
