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

Calliope is a framework to develop energy system models, with a focus on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data). Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents. In an optional operational it can also test a pre-defined system under different operational conditions.

A Calliope model consists of a collection of text files (in YAML and CSV formats) that fully define a model, with details on technologies, locations, resource potentials, etc. Calliope takes these files, constructs an optimization problem, solves it, and reports back results. Results can be saved to CSV or NetCDF files for further processing, or analysed directly in Python through Python's extensive scientific data processing capabilities provided by libraries like [Pandas](http://pandas.pydata.org/) and [xarray](https://docs.xarray.dev/en/stable/).

Model results can be explored interactively with [Calligraph](https://calligraph.readthedocs.io/), our companion visualisation tool. Having some knowledge of the Python programming language helps when running Calliope, but is not a prerequisite.

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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/en/reference/emoji-key/)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sjpfenninger"><img src="https://avatars.githubusercontent.com/u/141709?v=4?s=100" width="100px;" alt="Stefan Pfenninger-Lee"/><br /><sub><b>Stefan Pfenninger-Lee</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=sjpfenninger" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=sjpfenninger" title="Documentation">📖</a> <a href="#ideas-sjpfenninger" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Asjpfenninger" title="Bug reports">🐛</a> <a href="#fundingFinding-sjpfenninger" title="Funding Finding">🔍</a> <a href="#projectManagement-sjpfenninger" title="Project Management">📆</a> <a href="#promotion-sjpfenninger" title="Promotion">📣</a> <a href="#maintenance-sjpfenninger" title="Maintenance">🚧</a> <a href="#question-sjpfenninger" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/brynpickering"><img src="https://avatars.githubusercontent.com/u/17178478?v=4?s=100" width="100px;" alt="Bryn Pickering"/><br /><sub><b>Bryn Pickering</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=brynpickering" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=brynpickering" title="Documentation">📖</a> <a href="#ideas-brynpickering" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Abrynpickering" title="Bug reports">🐛</a> <a href="#fundingFinding-brynpickering" title="Funding Finding">🔍</a> <a href="#projectManagement-brynpickering" title="Project Management">📆</a> <a href="#promotion-brynpickering" title="Promotion">📣</a> <a href="#maintenance-brynpickering" title="Maintenance">🚧</a> <a href="#question-brynpickering" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://orcid.org/0000-0003-2288-6423"><img src="https://avatars.githubusercontent.com/u/72193617?v=4?s=100" width="100px;" alt="Ivan Ruiz Manuel"/><br /><sub><b>Ivan Ruiz Manuel</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=irm-codebase" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=irm-codebase" title="Documentation">📖</a> <a href="#ideas-irm-codebase" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Airm-codebase" title="Bug reports">🐛</a> <a href="#promotion-irm-codebase" title="Promotion">📣</a> <a href="#maintenance-irm-codebase" title="Maintenance">🚧</a> <a href="#question-irm-codebase" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://cp.ethz.ch/people/person-detail.tim-troendle.html"><img src="https://avatars.githubusercontent.com/u/3090386?v=4?s=100" width="100px;" alt="Tim Tröndle"/><br /><sub><b>Tim Tröndle</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=timtroendle" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=timtroendle" title="Documentation">📖</a> <a href="#ideas-timtroendle" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Atimtroendle" title="Bug reports">🐛</a> <a href="#promotion-timtroendle" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FLomb"><img src="https://avatars.githubusercontent.com/u/26432077?v=4?s=100" width="100px;" alt="Francesco Lombardi"/><br /><sub><b>Francesco Lombardi</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=FLomb" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=FLomb" title="Documentation">📖</a> <a href="#ideas-FLomb" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3AFLomb" title="Bug reports">🐛</a> <a href="#promotion-FLomb" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jnnr"><img src="https://avatars.githubusercontent.com/u/32454596?v=4?s=100" width="100px;" alt="Jann Launer"/><br /><sub><b>Jann Launer</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=jnnr" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/commits?author=jnnr" title="Documentation">📖</a> <a href="#ideas-jnnr" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajnnr" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ahilbers.github.io/"><img src="https://avatars.githubusercontent.com/u/31656517?v=4?s=100" width="100px;" alt="Adriaan Hilbers"/><br /><sub><b>Adriaan Hilbers</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=ahilbers" title="Code">💻</a> <a href="#ideas-ahilbers" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Aahilbers" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tomdeallycat"><img src="https://avatars.githubusercontent.com/u/50217534?v=4?s=100" width="100px;" alt="Tom Harris"/><br /><sub><b>Tom Harris</b></sub></a><br /><a href="#ideas-tomdeallycat" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Atomdeallycat" title="Bug reports">🐛</a> <a href="#promotion-tomdeallycat" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FraSanvit"><img src="https://avatars.githubusercontent.com/u/68587472?v=4?s=100" width="100px;" alt="Francesco Sanvito"/><br /><sub><b>Francesco Sanvito</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=FraSanvit" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3AFraSanvit" title="Bug reports">🐛</a> <a href="#promotion-FraSanvit" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sstroemer"><img src="https://avatars.githubusercontent.com/u/8915976?v=4?s=100" width="100px;" alt="Stefan Strömer"/><br /><sub><b>Stefan Strömer</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=sstroemer" title="Code">💻</a> <a href="#ideas-sstroemer" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Asstroemer" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arnaud-leroy"><img src="https://avatars.githubusercontent.com/u/29625919?v=4?s=100" width="100px;" alt="arnaud-leroy"/><br /><sub><b>arnaud-leroy</b></sub></a><br /><a href="#ideas-arnaud-leroy" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Aarnaud-leroy" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.strath.ac.uk/staff/hawkergraemedr/"><img src="https://avatars.githubusercontent.com/u/26121052?v=4?s=100" width="100px;" alt="Graeme Hawker"/><br /><sub><b>Graeme Hawker</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=GraemeHawker" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3AGraemeHawker" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://mammouth.ai/"><img src="https://avatars.githubusercontent.com/u/16239564?v=4?s=100" width="100px;" alt="Martial Garchery"/><br /><sub><b>Martial Garchery</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=mlgarchery" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Amlgarchery" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/smorgenthaler"><img src="https://avatars.githubusercontent.com/u/41112077?v=4?s=100" width="100px;" alt="smorgenthaler"/><br /><sub><b>smorgenthaler</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=smorgenthaler" title="Code">💻</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Asmorgenthaler" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mohammadamint"><img src="https://avatars.githubusercontent.com/u/50955527?v=4?s=100" width="100px;" alt="Mohammad Amin Tahavori"/><br /><sub><b>Mohammad Amin Tahavori</b></sub></a><br /><a href="#ideas-mohammadamint" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Amohammadamint" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/dominik-franjo-dominkovic"><img src="https://avatars.githubusercontent.com/u/33812038?v=4?s=100" width="100px;" alt="Dodo"/><br /><sub><b>Dodo</b></sub></a><br /><a href="#ideas-CROdominik" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3ACROdominik" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lblabr"><img src="https://avatars.githubusercontent.com/u/5998943?v=4?s=100" width="100px;" alt="lblabr"/><br /><sub><b>lblabr</b></sub></a><br /><a href="#ideas-lblabr" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Alblabr" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jamesfallon.eu"><img src="https://avatars.githubusercontent.com/u/2388576?v=4?s=100" width="100px;" alt="James Fallon"/><br /><sub><b>James Fallon</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajfallon1997" title="Bug reports">🐛</a> <a href="#question-jfallon1997" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ramaroesilva"><img src="https://avatars.githubusercontent.com/u/52629579?v=4?s=100" width="100px;" alt="Rodrigo Amaro e Silva"/><br /><sub><b>Rodrigo Amaro e Silva</b></sub></a><br /><a href="#ideas-ramaroesilva" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Aramaroesilva" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jmorrisnrel"><img src="https://avatars.githubusercontent.com/u/90803675?v=4?s=100" width="100px;" alt="James Morris"/><br /><sub><b>James Morris</b></sub></a><br /><a href="#ideas-jmorrisnrel" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajmorrisnrel" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fvandebeek"><img src="https://avatars.githubusercontent.com/u/108682781?v=4?s=100" width="100px;" alt="fvandebeek"/><br /><sub><b>fvandebeek</b></sub></a><br /><a href="#ideas-fvandebeek" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Afvandebeek" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yiqiaowang-arch"><img src="https://avatars.githubusercontent.com/u/28997207?v=4?s=100" width="100px;" alt="Yiqiao Wang"/><br /><sub><b>Yiqiao Wang</b></sub></a><br /><a href="#ideas-yiqiaowang-arch" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Ayiqiaowang-arch" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hiddegrootes"><img src="https://avatars.githubusercontent.com/u/151831022?v=4?s=100" width="100px;" alt="hiddegrootes"/><br /><sub><b>hiddegrootes</b></sub></a><br /><a href="#ideas-hiddegrootes" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Ahiddegrootes" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tud-mchen6"><img src="https://avatars.githubusercontent.com/u/133768452?v=4?s=100" width="100px;" alt="mchen6"/><br /><sub><b>mchen6</b></sub></a><br /><a href="#ideas-tud-mchen6" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Atud-mchen6" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cpalazzi"><img src="https://avatars.githubusercontent.com/u/61022595?v=4?s=100" width="100px;" alt="cpalazzi"/><br /><sub><b>cpalazzi</b></sub></a><br /><a href="#ideas-cpalazzi" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/calliope-project/calliope/issues?q=author%3Acpalazzi" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thormeyc"><img src="https://avatars.githubusercontent.com/u/15925225?v=4?s=100" width="100px;" alt="thormeyc"/><br /><sub><b>thormeyc</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Athormeyc" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/csv2000"><img src="https://avatars.githubusercontent.com/u/5485494?v=4?s=100" width="100px;" alt="Vijay C S"/><br /><sub><b>Vijay C S</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Acsv2000" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mkoehme"><img src="https://avatars.githubusercontent.com/u/29919988?v=4?s=100" width="100px;" alt="mkoehme"/><br /><sub><b>mkoehme</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Amkoehme" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/coroa"><img src="https://avatars.githubusercontent.com/u/2552981?v=4?s=100" width="100px;" alt="Jonas Hörsch"/><br /><sub><b>Jonas Hörsch</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Acoroa" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/louischaman"><img src="https://avatars.githubusercontent.com/u/8059916?v=4?s=100" width="100px;" alt="louischaman"/><br /><sub><b>louischaman</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Alouischaman" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/miraStud"><img src="https://avatars.githubusercontent.com/u/33147228?v=4?s=100" width="100px;" alt="miraStud"/><br /><sub><b>miraStud</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3AmiraStud" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/abart89"><img src="https://avatars.githubusercontent.com/u/25038235?v=4?s=100" width="100px;" alt="Andrea B."/><br /><sub><b>Andrea B.</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aabart89" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/katrinleinweber"><img src="https://avatars.githubusercontent.com/u/9948149?v=4?s=100" width="100px;" alt="Katrin Leinweber"/><br /><sub><b>Katrin Leinweber</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=katrinleinweber" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/mdoucet/"><img src="https://avatars.githubusercontent.com/u/1108748?v=4?s=100" width="100px;" alt="Mat Doucet"/><br /><sub><b>Mat Doucet</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Amdoucet" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AlexandreLab"><img src="https://avatars.githubusercontent.com/u/20833629?v=4?s=100" width="100px;" alt="AlexandreLab"/><br /><sub><b>AlexandreLab</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3AAlexandreLab" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://jedbrown.org"><img src="https://avatars.githubusercontent.com/u/3303?v=4?s=100" width="100px;" alt="Jed Brown"/><br /><sub><b>Jed Brown</b></sub></a><br /><a href="#ideas-jedbrown" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GiorgioBalestrieri"><img src="https://avatars.githubusercontent.com/u/17710158?v=4?s=100" width="100px;" alt="Giorgio Balestrieri"/><br /><sub><b>Giorgio Balestrieri</b></sub></a><br /><a href="#ideas-GiorgioBalestrieri" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alexsescu"><img src="https://avatars.githubusercontent.com/u/46405109?v=4?s=100" width="100px;" alt="alex"/><br /><sub><b>alex</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aalexsescu" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/b-jesse"><img src="https://avatars.githubusercontent.com/u/43567526?v=4?s=100" width="100px;" alt="b-jesse"/><br /><sub><b>b-jesse</b></sub></a><br /><a href="#ideas-b-jesse" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/darlainedeme"><img src="https://avatars.githubusercontent.com/u/45342315?v=4?s=100" width="100px;" alt="Darlain Edeme"/><br /><sub><b>Darlain Edeme</b></sub></a><br /><a href="#ideas-darlainedeme" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ewanfrostpennington"><img src="https://avatars.githubusercontent.com/u/33542881?v=4?s=100" width="100px;" alt="Ewan Frost-Pennington"/><br /><sub><b>Ewan Frost-Pennington</b></sub></a><br /><a href="#question-Ewanfrostpennington" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/brmanuel"><img src="https://avatars.githubusercontent.com/u/22857883?v=4?s=100" width="100px;" alt="brmanuel"/><br /><sub><b>brmanuel</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=brmanuel" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/suvayu"><img src="https://avatars.githubusercontent.com/u/229540?v=4?s=100" width="100px;" alt="Suvayu Ali"/><br /><sub><b>Suvayu Ali</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=suvayu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GlennCeusters"><img src="https://avatars.githubusercontent.com/u/29844834?v=4?s=100" width="100px;" alt="GlennCeusters"/><br /><sub><b>GlennCeusters</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3AGlennCeusters" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FebinKa"><img src="https://avatars.githubusercontent.com/u/55534006?v=4?s=100" width="100px;" alt="Febin Kachirayil"/><br /><sub><b>Febin Kachirayil</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3AFebinKa" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alicestamp"><img src="https://avatars.githubusercontent.com/u/57448730?v=4?s=100" width="100px;" alt="alicestamp"/><br /><sub><b>alicestamp</b></sub></a><br /><a href="#ideas-alicestamp" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leonardgoeke"><img src="https://avatars.githubusercontent.com/u/55208856?v=4?s=100" width="100px;" alt="leonardgoeke"/><br /><sub><b>leonardgoeke</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aleonardgoeke" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bbrannon4"><img src="https://avatars.githubusercontent.com/u/3052661?v=4?s=100" width="100px;" alt="bbrannon4"/><br /><sub><b>bbrannon4</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Abbrannon4" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Yannickvtil"><img src="https://avatars.githubusercontent.com/u/39329121?v=4?s=100" width="100px;" alt="yannick van til"/><br /><sub><b>yannick van til</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3AYannickvtil" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/namosata"><img src="https://avatars.githubusercontent.com/u/79442058?v=4?s=100" width="100px;" alt="namosata"/><br /><sub><b>namosata</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Anamosata" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Lingkangjin"><img src="https://avatars.githubusercontent.com/u/70599423?v=4?s=100" width="100px;" alt="Lingkang Jin"/><br /><sub><b>Lingkang Jin</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3ALingkangjin" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://121gigawatts.org"><img src="https://avatars.githubusercontent.com/u/5569474?v=4?s=100" width="100px;" alt="Zoltán Marić"/><br /><sub><b>Zoltán Marić</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Azoltanmaric" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/saim14"><img src="https://avatars.githubusercontent.com/u/41085427?v=4?s=100" width="100px;" alt="Saim Islam"/><br /><sub><b>Saim Islam</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Asaim14" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cristinaantonini"><img src="https://avatars.githubusercontent.com/u/61692338?v=4?s=100" width="100px;" alt="cristinaantonini"/><br /><sub><b>cristinaantonini</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Acristinaantonini" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adrienmellot"><img src="https://avatars.githubusercontent.com/u/97834931?v=4?s=100" width="100px;" alt="Adrien Mellot"/><br /><sub><b>Adrien Mellot</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aadrienmellot" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ollie-bell"><img src="https://avatars.githubusercontent.com/u/56110893?v=4?s=100" width="100px;" alt="ollie-bell"/><br /><sub><b>ollie-bell</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aollie-bell" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SisiLimperatrice"><img src="https://avatars.githubusercontent.com/u/91324429?v=4?s=100" width="100px;" alt="SisiLimperatrice"/><br /><sub><b>SisiLimperatrice</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3ASisiLimperatrice" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.svrijn.nl"><img src="https://avatars.githubusercontent.com/u/8833517?v=4?s=100" width="100px;" alt="Sander van Rijn"/><br /><sub><b>Sander van Rijn</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Asjvrijn" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/maurerle"><img src="https://avatars.githubusercontent.com/u/25026204?v=4?s=100" width="100px;" alt="Florian Maurer"/><br /><sub><b>Florian Maurer</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Amaurerle" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jsejdija"><img src="https://avatars.githubusercontent.com/u/124056869?v=4?s=100" width="100px;" alt="jsejdija"/><br /><sub><b>jsejdija</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajsejdija" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gnawin"><img src="https://avatars.githubusercontent.com/u/125902905?v=4?s=100" width="100px;" alt="Ni Wang"/><br /><sub><b>Ni Wang</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Agnawin" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jgu2"><img src="https://avatars.githubusercontent.com/u/50031970?v=4?s=100" width="100px;" alt="jgu2@nlr"/><br /><sub><b>jgu2@nlr</b></sub></a><br /><a href="#ideas-jgu2" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eeroinkeri"><img src="https://avatars.githubusercontent.com/u/70897930?v=4?s=100" width="100px;" alt="eeroinkeri"/><br /><sub><b>eeroinkeri</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aeeroinkeri" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Longquan-Li"><img src="https://avatars.githubusercontent.com/u/184640993?v=4?s=100" width="100px;" alt="Longquan-Li"/><br /><sub><b>Longquan-Li</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3ALongquan-Li" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jeisenman23"><img src="https://avatars.githubusercontent.com/u/157855136?v=4?s=100" width="100px;" alt="jeisenman23"/><br /><sub><b>jeisenman23</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajeisenman23" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ddahawkins-TUDelft"><img src="https://avatars.githubusercontent.com/u/181712209?v=4?s=100" width="100px;" alt="ddahawkins-TUDelft"/><br /><sub><b>ddahawkins-TUDelft</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Addahawkins-TUDelft" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/antoniodepadova"><img src="https://avatars.githubusercontent.com/u/152521975?v=4?s=100" width="100px;" alt="antoniodepadova"/><br /><sub><b>antoniodepadova</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aantoniodepadova" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/omahs"><img src="https://avatars.githubusercontent.com/u/73983677?v=4?s=100" width="100px;" alt="omahs"/><br /><sub><b>omahs</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=omahs" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jaakkohypi"><img src="https://avatars.githubusercontent.com/u/177741871?v=4?s=100" width="100px;" alt="jaakkohypi"/><br /><sub><b>jaakkohypi</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Ajaakkohypi" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/De-Hav"><img src="https://avatars.githubusercontent.com/u/197702496?v=4?s=100" width="100px;" alt="Regan Scott"/><br /><sub><b>Regan Scott</b></sub></a><br /><a href="#ideas-De-Hav" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://portalrecerca.uab.cat/en/persons/alexander-de-tom%C3%A1s-pascual"><img src="https://avatars.githubusercontent.com/u/79085248?v=4?s=100" width="100px;" alt="Alexander de Tomás Pascual"/><br /><sub><b>Alexander de Tomás Pascual</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3ALexPascal" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NiklasDenter"><img src="https://avatars.githubusercontent.com/u/184489280?v=4?s=100" width="100px;" alt="Niklas Denter"/><br /><sub><b>Niklas Denter</b></sub></a><br /><a href="#ideas-NiklasDenter" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TimothydW"><img src="https://avatars.githubusercontent.com/u/151521499?v=4?s=100" width="100px;" alt="TimothydW"/><br /><sub><b>TimothydW</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3ATimothydW" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joeseakinsarup"><img src="https://avatars.githubusercontent.com/u/103901492?v=4?s=100" width="100px;" alt="joeseakinsarup"/><br /><sub><b>joeseakinsarup</b></sub></a><br /><a href="#ideas-joeseakinsarup" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mkarmellos"><img src="https://avatars.githubusercontent.com/u/148218983?v=4?s=100" width="100px;" alt="mkarmellos"/><br /><sub><b>mkarmellos</b></sub></a><br /><a href="#ideas-mkarmellos" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/anedelcu2002"><img src="https://avatars.githubusercontent.com/u/127727911?v=4?s=100" width="100px;" alt="Alex Nedelcu"/><br /><sub><b>Alex Nedelcu</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Aanedelcu2002" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://geiser.cloud"><img src="https://avatars.githubusercontent.com/u/9169332?v=4?s=100" width="100px;" alt="Sergio Fernández"/><br /><sub><b>Sergio Fernández</b></sub></a><br /><a href="#promotion-GeiserX" title="Promotion">📣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kevin-moy-HSEO"><img src="https://avatars.githubusercontent.com/u/195054128?v=4?s=100" width="100px;" alt="Kevin Moy"/><br /><sub><b>Kevin Moy</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/issues?q=author%3Akevin-moy-HSEO" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marijacveevska"><img src="https://avatars.githubusercontent.com/u/94995858?v=4?s=100" width="100px;" alt="Marija Cveevska"/><br /><sub><b>Marija Cveevska</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=marijacveevska" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sayportfolio.vercel.app/"><img src="https://avatars.githubusercontent.com/u/240962040?v=4?s=100" width="100px;" alt="Sai Asish Y"/><br /><sub><b>Sai Asish Y</b></sub></a><br /><a href="https://github.com/calliope-project/calliope/commits?author=SAY-5" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
