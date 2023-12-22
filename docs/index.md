# Calliope: energy system modelling made simple

!!! note

    This is the documentation for version {{ calliope_version }} ([Version history](version_history.md)).
    See the [main project website at www.callio.pe](https://www.callio.pe/) for more general information, including a gallery of models built with Calliope, and other useful information.

Calliope focuses on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).
Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents.
In an optional operational mode it can also test a pre-defined system under different operational conditions.
Calliope's built-in tools allow interactive exploration of results.

[TODO: PLOT HERE]

A model based on Calliope consists of a collection of text files (in YAML and CSV formats) that define the technologies, locations and resource potentials.
Calliope takes these files, constructs an optimisation problem, solves it, and reports results in the form of [xarray Datasets](https://docs.xarray.dev/en/v2022.03.0/user-guide/data-structures.html#dataset) which in turn can easily be converted into [Pandas data structures](https://pandas.pydata.org/pandas-docs/version/1.5/user_guide/dsintro.html#dsintro) for easy analysis with Calliope's built-in tools or the standard Python data analysis stack.

Calliope is developed in the open [on GitHub](https://github.com/calliope-project/calliope) and contributions are very welcome (see TODO: LINK:`user/develop`).

Key features of Calliope include:

* Model specification in an easy-to-read and machine-processable YAML format
* Generic technology definition allows modelling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: read time series with arbitrary resolution
* Able to run on high-performance computing (HPC) clusters
* Uses a state-of-the-art Python toolchain based on [Pyomo](https://pyomo.readthedocs.io/en/stable/), [xarray](https://docs.xarray.dev/en/stable/), and [Pandas](https://pandas.pydata.org/)
* Freely available under the Apache 2.0 license

## License

Copyright since 2013 Calliope contributors listed in AUTHORS

Licensed under the Apache License, Version 2.0 (the "License"); you
may not use this file except in compliance with the License. You may
obtain a copy of the License at

<https://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
