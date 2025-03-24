[![GitHub Discussions](https://img.shields.io/github/discussions/calliope-project/calliope)](https://github.com/calliope-project/calliope/discussions)
[![Main branch build status](https://github.com/calliope-project/calliope/actions/workflows/commit-ci.yml/badge.svg?branch=main)](https://github.com/calliope-project/calliope/actions/workflows/commit-ci.yml)
[![Documentation build status](https://img.shields.io/readthedocs/calliope.svg?version=latest)](https://readthedocs.org/projects/calliope/builds/)
[![Test coverage](https://codecov.io/gh/calliope-project/calliope/graph/badge.svg?token=UM542yaYrh)](https://codecov.io/gh/calliope-project/calliope)
[![PyPI version](https://img.shields.io/pypi/v/calliope.svg)](https://pypi.python.org/pypi/calliope)
[![Anaconda.org/conda-forge version](https://img.shields.io/conda/vn/conda-forge/calliope.svg?label=conda)](https://anaconda.org/conda-forge/calliope)
[![JOSS DOI](https://img.shields.io/badge/JOSS-10.21105/joss.00825-green.svg)](https://doi.org/10.21105/joss.00825)

---

<img src="https://raw.githubusercontent.com/calliope-project/calliope/main/docs/img/logo.png" width="364">

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

A Calliope model consists of a collection of text files (in YAML and CSV formats) that fully define a model, with details on technologies, locations, resource potentials, etc. Calliope takes these files, constructs an optimization problem, solves it, and reports back results. Results can be saved to CSV or NetCDF files for further processing, or analysed directly in Python through Python's extensive scientific data processing capabilities provided by libraries like [Pandas](http://pandas.pydata.org/) and [xarray](https://docs.xarray.dev/en/stable/).

Calliope comes with several built-in analysis and visualisation tools. Having some knowledge of the Python programming language helps when running Calliope and using these tools, but is not a prerequisite.

## Quick start

Calliope can run on Windows, macOS and Linux. Installing it is quickest with the `mamba` package manager by running a single command: `mamba create -n calliope -c conda-forge conda-forge/label/calliope_dev::calliope`.

See the documentation for more [information on installing](https://calliope.readthedocs.io/en/stable/user/installation.html).

Several easy to understand example models are [included with Calliope](https://github.com/calliope-project/calliope/tree/main/src/calliope/example_models) and accessible through the `calliope.examples` submodule.

The [tutorials in the documentation run through these examples](https://calliope.readthedocs.io/en/stable/user/tutorials.html). A good place to start is to look at these tutorials to get a feel for how Calliope works, and then to read the "Introduction", "Building a model", "Running a model", and "Analysing a model" sections in the online documentation.

More fully-featured examples that have been used in peer-reviewed scientific publications are available in our [model gallery](https://www.callio.pe/research/#models).

## Documentation

Documentation is available on [Read the Docs](https://calliope.readthedocs.io/en/stable/).

## Contributing

See our documentation for more on how to [contribute to Calliope](http://calliope.readthedocs.io/en/latest/contributing/).

## What's new

See changes made in recent versions in the [changelog](https://github.com/calliope-project/calliope/blob/main/CHANGELOG.md).

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
