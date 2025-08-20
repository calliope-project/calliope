# Calliope: energy system modelling made simple

!!! warning

    Calliope version 0.7 is available as a pre-release with the purpose of gathering feedback from users.
    To install the pre-release as a user:

    ```shell
    mamba create -n calliope -c conda-forge conda-forge/label/calliope_dev::calliope
    ```

    Visualisation of results has moved to a separate tool, [Calligraph](https://calligraph.readthedocs.io/).

    To see a full list of changes, read our [page on migrating between v0.6 and v0.7](migrating.md).

    If you want to install the most recent _stable_ version of Calliope, see our [v0.6.10 documentation](https://calliope.readthedocs.io/en/v0.6.10/).

!!! note

    This is the documentation for version {{ calliope_version }} ([version history](version_history.md)).
    See the [main project website at www.callio.pe](https://www.callio.pe/) for more general information, including a gallery of models built with Calliope, and other useful information.

Calliope focuses on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).
Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents.
In an optional operational mode it can also test a pre-defined system under different operational conditions.
The [Calligraph companion tool](https://calligraph.readthedocs.io/) allows interactive exploration of results.

Visualising with Calligraph:

<video controls>
    <source src="https://spontaneous-choux-e05fa1.netlify.app/calligraph.mp4" type="video/mp4">
</video>

Visualising with Plotly:

<object type="text/html" data="img/plotly_frontpage_timeseries.html" width="100%" height="400px"></object>

A model based on Calliope consists of a collection of text files (in YAML and CSV formats) that define the technologies, locations and resource potentials.
Calliope takes these files, constructs an optimisation problem, solves it, and reports results in the form of [xarray Datasets](https://docs.xarray.dev/en/v2022.03.0/user-guide/data-structures.html#dataset) which in turn can easily be converted into [Pandas data structures](https://pandas.pydata.org/pandas-docs/version/1.5/user_guide/dsintro.html#dsintro) for easy analysis with Calliope's built-in tools or the standard Python data analysis stack.

Calliope is developed in the open [on GitHub](https://github.com/calliope-project/calliope) and contributions are very welcome (see the [section on contributing](contributing.md)).

Key features of Calliope include:

* Model specification in an easy-to-read and machine-processable YAML format
* Generic technology definition allows modelling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: read time series with arbitrary resolution
* Able to run on high-performance computing (HPC) clusters
* Uses a state-of-the-art Python toolchain based on [Pyomo](https://pyomo.readthedocs.io/en/stable/), [xarray](https://docs.xarray.dev/en/stable/), and [Pandas](https://pandas.pydata.org/)
* Freely available under the Apache 2.0 license

## Acknowledgements

See the [callio.pe project website](https://www.callio.pe/partners-and-team/) for current and past team members and acknowledgements.

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

## Citing Calliope

Calliope is [published in the Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.00825).
We encourage you to use this academic reference.
