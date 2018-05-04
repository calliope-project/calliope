---
title: 'Calliope: a multi-scale energy systems modelling framework'
tags:
  - energy
  - optimisation
  - python
authors:
 - name: Stefan Pfenninger
   orcid: 0000-0002-8420-9498
   affiliation: 1
 - name: Bryn Pickering
   orcid: 0000-0003-4044-6587
   affiliation: 2
affiliations:
 - name: Department of Environmental Systems Science, ETH Zürich
   index: 1
 - name: Department of Engineering, University of Cambridge
   index: 2
date: 30 April 2018
bibliography: paper.bib
---

# Summary

Energy system models create coherent quantitative descriptions of how energy is converted, transported, and consumed, at scales ranging from urban districts to entire continents. Formulating such models as optimisation problems allows a modeller to asses the effect of constraints, such as limited land availability for wind power deployment, the cost of battery electricity storage, or the elimination of fossil fuels from a country or a city, on the feasibility or cost of the modelled system. These models are particularly important in planning and policy-making for the transformation of the global energy system to address climate change.

Calliope is a framework to build energy system models, designed to analyse systems with arbitrarily high spatial and temporal resolution, with a scale-agnostic mathematical formulation permitting analyses ranging from single urban districts to countries and continents. Its formulation of energy system components was influenced by the power nodes modelling framework by Heussen et al. [@Heussen2010], but generalised to consider energy carriers other than electricity. Calliope's key features include the ability to handle high spatial and temporal resolution and to easily run on high-performance computing systems. Its design cleanly separates the general framework (code) from the problem-specific model (data). It provides both a command-line interface and an API for programmatic use, to be useful both for users experienced with Python and those with no Python knowledge.

A Calliope model consists of a collection of ``YAML`` and ``CSV`` files that define technologies, locations, links between locations, resource potentials, and other constraints. Calliope takes these files, constructs an optimisation problem, solves it, and reports results in the form of ``xarray`` [@xarray] Datasets, which can easily be saved to NetCDF files for further processing. It uses Pyomo [@pyomo] as a backend to interface with both open and commercial solvers, currently handling linear and mixed-integer problems, although nonlinear components could be implemented if necessary for new kinds of problems. Calliope's built-in tools allow interactive exploration of results using Plotly [@plotly], as shown in Figure 1.

![Example time series visualisation of aggregated generation decisions at hourly time scale from a national-scale model of the UK power system, created with the Plotly-based visualisation tools in Calliope.](timeseries.pdf)

Calliope has been used in various studies, for example, analyses of the national-scale power systems in Britain [@Pfenninger2015_1] and South Africa [@Pfenninger2015_2], and in methodological development for piecewise linearisation of characteristic technology performance curves for district-scale energy system analysis [@Pickering2017]. Ongoing research projects using Calliope include the effect of increased resilience to uncertain future demand and the interaction between local and national actors in the clean energy transition.

Calliope is developed in the open on GitHub [@CalliopeGitHub] and each release is archived on Zenodo [@CalliopeZenodo].

# Acknowledgements

The authors acknowledge funding via the European Research Council (grant StG 2012-313553), the Grantham Institute for Climate Change at Imperial College London, and the Engineering and Physical Sciences Research Council (ref EP/L016095/1).

# References
