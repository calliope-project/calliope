
============
Introduction
============

The basic process of modelling with Calliope is based on three steps:

1. Create a model from scratch or by adjusting an existing model (:doc:`building`)
2. Run your model (:doc:`running`)
3. Analyse and visualise model results (:doc:`analysing`)

--------------------
Energy system models
--------------------

Energy system models allow analysts to form internally coherent scenarios of how energy is extracted, converted, transported, and used, and how these processes might change in the future. These models have been gaining renewed importance as methods to help navigate the climate policy-driven transformation of the energy system.

Calliope is an attempt to design an energy system model from the ground of up with specific design goals in mind (see below). Therefore, the model approach and data format layout may be different from approaches used in other models. The design of the nodes approach used in Calliope was influenced by the power nodes modelling framework by [Heussen2010]_, but Calliope is different from traditional power system modelling tools, and does not provide features such as power flow analysis.

Calliope was designed to address questions around the transition to renewable energy, so there are tools that are likely to be more suitable for other types of questions. In particular, the following related energy modelling systems are available under open source or free software licenses:

* `SWITCH <https://switch-model.org/>`_: A power system model focused on renewables integration, using multi-stage stochastic linear optimisation, as well as hourly resource potential and demand data. Written in the commercial AMPL language and GPL-licensed [Fripp2012]_.
* `Temoa <https://temoacloud.com/>`_: An energy system model with multi-stage stochastic optimisation functionality which can be deployed to computing clusters, to address parametric uncertainty. Written in Python/Pyomo and AGPL-licensed [Hunter2013]_.
* `OSeMOSYS <http://www.osemosys.org/>`_: A simplified energy system model similar to the MARKAL/TIMES model families, which can be used as a stand-alone tool or integrated in the `LEAP energy model <https://leap.sei.org/>`_. Written in GLPK, a free subset of the commercial AMPL language, and Apache 2.0-licensed [Howells2011]_.

Additional energy models that are partially or fully open can be found on the `Open Energy Modelling Initiative's wiki <https://wiki.openmod-initiative.org/wiki/Model_fact_sheets>`_.

.. _rationale:

---------
Rationale
---------

Calliope was designed with the following goals in mind:

* Designed from the ground up to analyze energy systems with high shares of renewable energy or other variable generation
* Formulated to allow arbitrary spatial and temporal resolution, and equipped with the necessary tools to deal with time series input data
* Allow easy separation of model code and data, and modular extensibility of model code
* Make models easily modifiable, archiveable and auditable (e.g. in a Git repository), by using well-defined and human-readable text formats
* Simplify the definition and deployment of large numbers of model runs to high-performance computing clusters
* Able to run stand-alone from the command-line, but also provide an API for programmatic access and embedding in larger analyses
* Be a first-class citizen of the Python world (installable with ``conda`` and ``pip``, with properly documented and tested code that mostly conforms to PEP8)
* Have a free and open-source code base under a permissive license

---------------
Acknowledgments
---------------

Development has been partially funded by several grants throughout throughout the years. We would particularly like to acknowledge the following:

* The `Grantham Institute <https://www.imperial.ac.uk/grantham>`_ at Imperial College London.
* the European Institute of Innovation & Technology's `Climate-KIC program <https://www.climate-kic.org>`_.
* `Engineering and Physical Sciences Research Council <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/L016095/1>`_, reference number: EP/L016095/1.
* `The Swiss Competence Center for Energy Research - Supply of Electricity (SCCER SoE) <http://sccer-soe.ch/en/home/>`_, contract number 1155002546.
* `Swiss Federal Office for Energy (SFOE) <https://www.bfe.admin.ch/bfe/en/home.html>`_, grant number SI/501768-01.
* `European Research Council <https://erc.europa.eu>`_ TRIPOD grant, grant agreement number 715132.
* `The SENTINEL project <https://sentinel.energy/>`_ of the European Union's Horizon 2020 research and innovation programme under grant agreement No 837089.

.. _license:

-------
License
-------

Calliope is released under the Apache 2.0 license, which is a permissive open-source license much like the MIT or BSD licenses. This means that Calliope can be incorporated in both commercial and non-commercial projects.

::

   Copyright since 2013 Calliope contributors listed in AUTHORS

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

----------
References
----------

.. [Fripp2012] Fripp, M., 2012. Switch: A Planning Tool for Power Systems with Large Shares of Intermittent Renewable Energy. Environ. Sci. Technol., 46(11), p.6371–6378. `DOI: 10.1021/es204645c <https://doi.org/10.1021/es204645c>`_
.. [Heussen2010] Heussen, K. et al., 2010. Energy storage in power system operation: The power nodes modeling framework. In Innovative Smart Grid Technologies Conference Europe (ISGT Europe), 2010 IEEE PES. pp. 1–8. `DOI: 10.1109/ISGTEUROPE.2010.5638865 <https://doi.org/10.1109/ISGTEUROPE.2010.5638865>`_
.. [Howells2011] Howells, M. et al., 2011. OSeMOSYS: The Open Source Energy Modeling System: An introduction to its ethos, structure and development. Energy Policy, 39(10), p.5850–5870. `DOI: 10.1016/j.enpol.2011.06.033 <https://doi.org/10.1016/j.enpol.2011.06.033>`_
.. [Hunter2013] Hunter, K., Sreepathi, S. & DeCarolis, J.F., 2013. Modeling for insight using Tools for Energy Model Optimization and Analysis (Temoa). Energy Economics, 40, p.339–349. `DOI: 10.1016/j.eneco.2013.07.014 <https://doi.org/10.1016/j.eneco.2013.07.014>`_

--------------------------------------
Citing Calliope in academic literature
--------------------------------------

Calliope is `published in the Journal of Open Source Software <https://joss.theoj.org/papers/10.21105/joss.00825>`_. We encourage you to use this academic reference.