# Calliope: energy system modelling made simple

!!! note

    This version of Calliope is available as a pre-release with the purpose of gathering feedback from users.
    To install the pre-release:

    ```shell
    mamba create -n calliope -c conda-forge conda-forge/label/calliope_dev::calliope
    ```

    To see a full list of changes, read our [page on migrating between v0.6 and v0.7](migrating.md).

    If you want to install the most recent _stable_ version of Calliope, see our [v0.6.10 documentation](https://calliope.readthedocs.io/en/v0.6.10/).

Calliope is an energy system modelling framework based on mathematical optimisation.
It is designed to formulate and solve typical problems from the energy field such as capacity expansion planning, economic dispatch, power market modelling and energy system modelling in general.
It is used in such roles by both commercial and research organisations.

Calliope focuses on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).
Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents.
In an optional operational mode it can also test a pre-defined system under different operational conditions.

This is the documentation for Calliope version {{ calliope_version }} ([version history](version_history.md)).
See the [main project website at www.callio.pe](https://www.callio.pe/) for more general information, including a gallery of models built with Calliope, and other useful information.

!!! tip "Where to start"

    If you are new to Calliope, we recommend that you first read through the [getting started guide](getting_started/concepts.md), then review the [examples and tutorials](examples/overview.md). The remaining sections primarily contain reference material that is most useful if you already know the basics and need to look up specifics.

A model based on Calliope consists of a collection of text files (in YAML and CSV formats) that define the technologies, locations and resource potentials.
Calliope takes these files, constructs an optimisation problem, solves it, and reports results in the form of [xarray Datasets](https://docs.xarray.dev/en/v2022.03.0/user-guide/data-structures.html#dataset) which in turn can easily be converted into [Pandas data structures](https://pandas.pydata.org/pandas-docs/version/1.5/user_guide/dsintro.html#dsintro) for further analysis.

<object type="text/html" data="img/plotly_frontpage_timeseries.html" width="100%" height="400px"></object>

Calliope is developed in the open [on GitHub](https://github.com/calliope-project/calliope) and contributions are very welcome (see the [section on contributing](contributing.md)).

Key features of Calliope include:

* Free and open-source (available under the Apache 2.0 license)
* Model specification in an easy-to-read and machine-processable YAML format
* Generic technology definition allows modelling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: read time series with arbitrary resolution
* Able to run on high-performance computing (HPC) clusters
* Uses a state-of-the-art Python toolchain based on [Pyomo](https://pyomo.readthedocs.io/en/stable/), [xarray](https://docs.xarray.dev/en/stable/), and [Pandas](https://pandas.pydata.org/)
* [Calligraph companion tool](https://calligraph.readthedocs.io/) for interactive exploration of results - see the example below:
<video controls>
    <source src="https://spontaneous-choux-e05fa1.netlify.app/calligraph.mp4" type="video/mp4">
</video>

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
