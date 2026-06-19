# Calliope: energy system modelling made simple

 This is the documentation for Calliope version {{ calliope_version }} ([version history](version_history.md)).

<div class="grid cards" markdown>

-   :fontawesome-solid-rocket:{ .lg .middle } __Starting quickly__

    * [Download and installation](installation.md): get Calliope installed and ready to use.
    * [Getting started](getting_started/concepts.md): read through this first to understand Calliope's basic concepts.
    * [Examples & tutorials](examples/index.md): these examples are best understood after you've gone through the basic concepts.

-   :fontawesome-solid-book:{ .lg .middle } __Going deeper__

    * [Building blocks](basic/index.md): more detailed explanation of all the building blocks that make up a Calliope model.
    * [How to](advanced/index.md): how to troubleshoot and access more advanced features like solver customisation or shadow prices.
    * [Math gallery](user_defined_math/examples/index.md): gallery of reusable, user-defined math for implementing advanced constraints.

-   :material-bookshelf:{ .lg .middle } __Reference__

    * [YAML, command line & schemas](reference/yaml.md): the YAML syntax, command-line interface, and configuration schemas.
    * [Python API](reference/api/model.md): reference documentation for the API.
    * [Built-in math](math/base.md): the base and other built-in math formulations.

-   :fontawesome-solid-signs-post:{ .lg .middle } __Other places to look__

    * [Migrating between versions](migrating.md): what changed between v0.6 and v0.7 and how to update your models.
    * [Contributing](contributing.md): how to contribute to Calliope's development.
    * [Version history](version_history.md): the full changelog.
    * Also see [www.callio.pe](https://www.callio.pe/) for more general information on the Calliope project.

</div>

## About Calliope

Calliope is an energy system modelling framework based on mathematical optimisation.
It is designed to formulate and solve typical problems from the energy field such as capacity expansion planning, economic dispatch, power market modelling and energy system modelling in general.
It is used in such roles by both commercial and research organisations.

Calliope focuses on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).
Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents.
In an optional operational mode it can also test a pre-defined system under different operational conditions.

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
