[![Zulip chat](https://img.shields.io/badge/chat-Zulip-blue?logo=zulip)](https://calliope-modelblocks.zulipchat.com/)
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

Calliope is a framework to develop energy system models, with a focus on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).
Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents.
In an optional operational mode it can also test a pre-defined system under different operational conditions.

A Calliope model consists of a collection of text files (in YAML and CSV formats) that fully define a model, with details on technologies, locations, resource potentials, etc.
Calliope takes these files, constructs an optimization problem, solves it, and reports back results.
Results can be saved to CSV or NetCDF files for further processing, or analysed directly in Python through Python's extensive scientific data processing capabilities provided by libraries like [Pandas](http://pandas.pydata.org/) and [xarray](https://docs.xarray.dev/en/stable/).

Model results can be explored interactively with [Calligraph](https://calligraph.readthedocs.io/), our companion visualisation tool.
Having some knowledge of the Python programming language helps when running Calliope, but is not a prerequisite.

## Quick start

Calliope can run on Windows, macOS and Linux.
It can be installed using `pixi` (`pixi add calliope`), `conda` (`conda install calliope`), or `uv` (`uv pip install calliope`).
For local development, use `pixi` as described in the [documentation](http://calliope.readthedocs.io/en/latest/contributing/).

See the documentation for more [information on installing](https://calliope.readthedocs.io/en/latest/user/installation.html).

Several easy to understand example models are [included with Calliope](https://github.com/calliope-project/calliope/tree/main/src/calliope/example_models) and accessible through the `calliope.examples` submodule.

The [tutorials in the documentation run through these examples](https://calliope.readthedocs.io/en/latest/examples/).
A good place to start is to look at these tutorials to get a feel for how Calliope works, and then to read the "Getting Started" pages in the [online documentation](https://calliope.readthedocs.io/en/latest/installation/).

More fully-featured examples that have been used in peer-reviewed scientific publications are available in our [model gallery](https://www.callio.pe/research/#models).

## Documentation

Documentation is available on [Read the Docs](https://calliope.readthedocs.io/en/latest/).

## Contributing

See our documentation for more on how to [contribute to Calliope](http://calliope.readthedocs.io/en/latest/contributing/).

## What's new

See changes made in recent versions in the [changelog](https://github.com/calliope-project/calliope/blob/main/CHANGELOG.md).

## Citing Calliope

If you use Calliope for academic work please cite:

Stefan Pfenninger and Bryn Pickering (2018).
Calliope: a multi-scale energy systems modelling framework. *Journal of Open Source Software*, 3(29), 825. [doi: 10.21105/joss.00825](https://doi.org/10.21105/joss.00825)

## License

Copyright since 2013 Calliope contributors listed in AUTHORS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/en/reference/emoji-key/)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sjpfenninger"><img src="https://avatars.githubusercontent.com/u/141709?v=4?s=100" width="100px;" alt="Stefan Pfenninger-Lee"/><br /><sub><b>Stefan Pfenninger-Lee</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=sjpfenninger" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=sjpfenninger" title="Documentation">📖</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Asjpfenninger" title="Bug reports">🐛</a> <a href="#fundingFinding-sjpfenninger" title="Funding Finding">🔍</a> <a href="#projectManagement-sjpfenninger" title="Project Management">📆</a> <a href="#promotion-sjpfenninger" title="Promotion">📣</a> <a href="#maintenance-sjpfenninger" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/brynpickering"><img src="https://avatars.githubusercontent.com/u/17178478?v=4?s=100" width="100px;" alt="Bryn Pickering"/><br /><sub><b>Bryn Pickering</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=brynpickering" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=brynpickering" title="Documentation">📖</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Abrynpickering" title="Bug reports">🐛</a> <a href="#fundingFinding-brynpickering" title="Funding Finding">🔍</a> <a href="#projectManagement-brynpickering" title="Project Management">📆</a> <a href="#promotion-brynpickering" title="Promotion">📣</a> <a href="#maintenance-brynpickering" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://orcid.org/0000-0003-2288-6423"><img src="https://avatars.githubusercontent.com/u/72193617?v=4?s=100" width="100px;" alt="Ivan Ruiz Manuel"/><br /><sub><b>Ivan Ruiz Manuel</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=irm-codebase" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=irm-codebase" title="Documentation">📖</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Airm-codebase" title="Bug reports">🐛</a> <a href="#promotion-irm-codebase" title="Promotion">📣</a> <a href="#maintenance-irm-codebase" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://cp.ethz.ch/people/person-detail.tim-troendle.html"><img src="https://avatars.githubusercontent.com/u/3090386?v=4?s=100" width="100px;" alt="Tim Tröndle"/><br /><sub><b>Tim Tröndle</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=timtroendle" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=timtroendle" title="Documentation">📖</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Atimtroendle" title="Bug reports">🐛</a> <a href="#promotion-timtroendle" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FLomb"><img src="https://avatars.githubusercontent.com/u/26432077?v=4?s=100" width="100px;" alt="Francesco Lombardi"/><br /><sub><b>Francesco Lombardi</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=FLomb" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=FLomb" title="Documentation">📖</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3AFLomb" title="Bug reports">🐛</a> <a href="#promotion-FLomb" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jnnr"><img src="https://avatars.githubusercontent.com/u/32454596?v=4?s=100" width="100px;" alt="Jann Launer"/><br /><sub><b>Jann Launer</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=jnnr" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajnnr" title="Bug reports">🐛</a> <a href="https://github.com/calliope-project/calliope/commits?author=jnnr" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FraSanvit"><img src="https://avatars.githubusercontent.com/u/68587472?v=4?s=100" width="100px;" alt="Francesco Sanvito"/><br /><sub><b>Francesco Sanvito</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=FraSanvit" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3AFraSanvit" title="Bug reports">🐛</a> <a href="#promotion-FraSanvit" title="Promotion">📣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.strath.ac.uk/staff/hawkergraemedr/"><img src="https://avatars.githubusercontent.com/u/26121052?v=4?s=100" width="100px;" alt="Graeme Hawker"/><br /><sub><b>Graeme Hawker</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=GraemeHawker" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3AGraemeHawker" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://mammouth.ai/"><img src="https://avatars.githubusercontent.com/u/16239564?v=4?s=100" width="100px;" alt="Martial Garchery"/><br /><sub><b>Martial Garchery</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=mlgarchery" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Amlgarchery" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/smorgenthaler"><img src="https://avatars.githubusercontent.com/u/41112077?v=4?s=100" width="100px;" alt="smorgenthaler"/><br /><sub><b>smorgenthaler</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=smorgenthaler" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Asmorgenthaler" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ahilbers.github.io/"><img src="https://avatars.githubusercontent.com/u/31656517?v=4?s=100" width="100px;" alt="Adriaan Hilbers"/><br /><sub><b>Adriaan Hilbers</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=ahilbers" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Aahilbers" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sstroemer"><img src="https://avatars.githubusercontent.com/u/8915976?v=4?s=100" width="100px;" alt="Stefan Strömer"/><br /><sub><b>Stefan Strömer</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=sstroemer" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Asstroemer" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/katrinleinweber"><img src="https://avatars.githubusercontent.com/u/9948149?v=4?s=100" width="100px;" alt="Katrin Leinweber"/><br /><sub><b>Katrin Leinweber</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=katrinleinweber" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/brmanuel"><img src="https://avatars.githubusercontent.com/u/22857883?v=4?s=100" width="100px;" alt="brmanuel"/><br /><sub><b>brmanuel</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=brmanuel" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/suvayu"><img src="https://avatars.githubusercontent.com/u/229540?v=4?s=100" width="100px;" alt="Suvayu Ali"/><br /><sub><b>Suvayu Ali</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=suvayu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/omahs"><img src="https://avatars.githubusercontent.com/u/73983677?v=4?s=100" width="100px;" alt="omahs"/><br /><sub><b>omahs</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=omahs" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marijacveevska"><img src="https://avatars.githubusercontent.com/u/94995858?v=4?s=100" width="100px;" alt="Marija Cveevska"/><br /><sub><b>Marija Cveevska</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=marijacveevska" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sayportfolio.vercel.app/"><img src="https://avatars.githubusercontent.com/u/240962040?v=4?s=100" width="100px;" alt="Sai Asish Y"/><br /><sub><b>Sai Asish Y</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=SAY-5" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
Contributions of any kind welcome!
