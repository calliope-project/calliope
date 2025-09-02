# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope configuration definition."""

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field

from calliope.schemas.general import (
    AttrStr,
    CalliopeBaseModel,
    CalliopeDictModel,
    NonEmptyList,
    UniqueList,
)

Mode = Literal["base", "operate", "spores"]

LOGGER = logging.getLogger(__name__)

SPORES_SCORING_OPTIONS = Literal[
    "integer", "relative_deployment", "random", "evolving_average"
]


class Subsets(CalliopeDictModel):
    """Dimension subsets."""

    root: dict[AttrStr, NonEmptyList[str | int | float] | None] = Field(
        default_factory=dict
    )


class Resamples(CalliopeDictModel):
    """Dimension resampling settings."""

    root: dict[AttrStr, str | None] = Field(default_factory=dict)


class MathPaths(CalliopeDictModel):
    """Math paths settings."""

    root: dict[AttrStr, Path] = Field(default_factory=dict)


class Init(CalliopeBaseModel):
    """All configuration options used when initialising a Calliope model."""

    model_config = {"title": "Model initialisation configuration"}
    name: str | None = None
    """Model name."""

    calliope_version: str | None = None
    """Calliope framework version this model is intended for."""

    broadcast_input_data: bool = Field(default=False)
    """
    If True, single data entries in YAML indexed data will be broadcast across all index items.
    Otherwise, the number of data entries needs to match the number of index items.
    Defaults to False to mitigate unexpected broadcasting when applying overrides.
    """

    subset: Subsets = Subsets()
    """
    Subset of timesteps as an two-element list giving the **inclusive** range.
    For example, ["2005-01", "2005-04"] will create a time subset from "2005-01-01 00:00:00" to "2005-04-31 23:59:59".

    Strings must be ISO8601-compatible, i.e. of the form `YYYY-mm-dd HH:MM:SS` (e.g, '2005-01 ', '2005-01-01', '2005-01-01 00:00', ...)
    """

    resample: Resamples = Resamples()
    """Setting to adjust datetime dimension resolution, e.g. '2h' for 2-hourly"""

    time_cluster: AttrStr | None = None
    """
    Setting to cluster the timeseries.
    Must reference the name of an input data array.
    """

    datetime_format: str = Field(default="ISO8601")
    """
    Timestamp format of all time series data with `datetime` dtype when read from file.
    'ISO8601' means '%Y-%m-%d %H:%M:%S'.
    """

    date_format: str = Field(default="ISO8601")
    """
    Datestamp format of all time series data with `date` dtype when read from file.
    'ISO8601' means '%Y-%m-%d'.
    """

    distance_unit: Literal["km", "m"] = Field(default="km")
    """
    Unit of transmission link `distance` (m - metres, km - kilometres).
    Automatically derived distances from lat/lon coordinates will be given in this unit.
    """

    mode: Mode = Field(default="base")
    """Mode in which to run the optimisation.
    Triggers additional processing and appends additional math formulations.
    Math order: base -> mode
    """

    extra_math: UniqueList[str] = Field(default_factory=list)
    """
    List of math entries to be applied on top of the `base` math and `mode` math.
    The list items must have been defined as keys in `math_paths` (see below).
    Math order: base -> mode -> extra
    """

    math_paths: MathPaths = MathPaths()
    """Dictionary with the names and paths of additional math files to add to the available math entries.
    Some math entry names are linked to specific functionality, so re-defining them here will overwrite the pre-defined math.:
    - `base`: replaces the pre-defined base math.
    - `milp`: replaces the mixed integer math.
    - `spores`/`operate`: replaces the respective pre-defined mode math.
    - `storage_inter_cluster`: replaces the pre-defined storage inter-cluster math.
    """

    pre_validate_math_strings: bool = Field(default=False)
    """
    If true, the Calliope math definition will be scanned for parsing errors at model initialisation,
    i.e., _before_ undertaking the much more expensive operation of building the optimisation problem.
    It is switched off by default to reduce overall build time.
    """


class BuildOperate(CalliopeBaseModel):
    """Operate mode configuration options used when building a Calliope optimisation problem (`calliope.Model.build`)."""

    model_config = {"title": "Model build operate mode configuration"}
    window: str = Field(default="24h")
    """
    Operate mode rolling `window`, given as a pandas frequency string.
    See [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases) for a list of frequency aliases.
    """

    horizon: str = Field(default="48h")
    """
    Operate mode rolling `horizon`, given as a pandas frequency string.
    See [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases) for a list of frequency aliases.
    Must be ≥ `window`
    """


class Build(CalliopeBaseModel):
    """Base configuration options used when building a Calliope optimisation problem (`calliope.Model.build`)."""

    model_config = {"title": "Model build configuration"}

    backend: Literal["pyomo", "gurobi"] = Field(default="pyomo")
    """Module with which to build the optimisation problem."""

    ensure_feasibility: bool = Field(default=False)
    """
    Whether to include decision variables in the model which will meet unmet demand or consume unused supply in the model so that the optimisation solves successfully.
    This should only be used as a debugging option (as any unmet demand/unused supply is a sign of improper model formulation).
    """

    objective: str = Field(default="min_cost_optimisation")
    """Name of internal objective function to use, from those defined in the pre-defined math and any applied additional math."""

    operate: BuildOperate = BuildOperate()
    """Operate mode specific configuration."""


class SolveSpores(CalliopeBaseModel):
    """SPORES configuration options used when solving a Calliope optimisation problem (`calliope.Model.solve`)."""

    model_config = {"title": "Model solve SPORES mode configuration"}

    scoring_algorithm: SPORES_SCORING_OPTIONS = "integer"
    """
    Algorithm to apply to update the SPORES score between iterations.
    For more information on each option, see [Lombardi et al. (2023)](https://doi.org/10.1016/j.apenergy.2023.121002).
    """

    number: int = Field(default=3)
    """SPORES mode number of iterations after the initial base run."""

    save_per_spore_path: Path | None = None
    """
    If None, the SPORES results will only be available in `calliope.Model.results` once all iterations (defined by `number`) have completed.
    If a path, as well as consolidating the SPORES results in `calliope.Model.results`, individual SPORES will be saved to file immediately after the iteration has completed.
    """

    use_latest_results: bool = Field(default=False)
    """
    If the model already contains `base` mode results, use them as the baseline results and start with SPORES iterations immediately.
    If the model already contains `spores` mode results, use the most recent results and continue with the remaining SPORES iterations immediately.
    """

    tracking_parameter: str | None = None
    """If given, an input parameter name with which to filter technologies for consideration in SPORES scoring."""

    score_threshold_factor: float = Field(default=0.1, ge=0)
    """A factor to apply to flow capacities above which they will increment the SPORES score.
    E.g., if the previous iteration flow capacity was `100` then, with a threshold value of 0.1,
    only capacities above `10` in the current iteration will cause the SPORES score to increase for that technology at that node.
    If, say, the current iteration's capacity is `8` then the SPORES score will not change for that technology so it will not be further penalised on the next iteration.
    """


class Solve(CalliopeBaseModel):
    """Base configuration options used when solving a Calliope optimisation problem (`calliope.Model.solve`)."""

    model_config = {"title": "Model Solve Configuration"}

    postprocessing_active: bool = True
    """If enabled, all active postprocessing functions will be run after the model solves."""

    save_logs: Path | None = None
    """If given, should be a path to a directory in which to save optimisation logs."""

    shadow_prices: UniqueList[str] = Field(default_factory=list)
    """Names of model constraints."""

    solver: str = Field(default="cbc")
    """Solver to use. Any solvers that have Pyomo interfaces can be used. Refer to the Pyomo documentation for the latest list."""

    solver_io: str | None = None
    """
    Some solvers have different interfaces that perform differently.
    For instance, setting `solver_io="python"` when using the solver `gurobi` tends to reduce the time to send the optimisation problem to the solver.
    """

    solver_options: dict = Field(default_factory=dict)
    """Any solver options, as key-value pairs, to pass to the chosen solver"""

    spores: SolveSpores = SolveSpores()
    """Spores configuration."""

    zero_threshold: float = Field(default=1e-10)
    """On postprocessing the optimisation results, values smaller than this threshold will be considered as optimisation artefacts and will be set to zero."""


class CalliopeConfig(CalliopeBaseModel):
    """Calliope configuration class."""

    model_config = {"title": "Model configuration schema"}

    init: Init = Init()
    build: Build = Build()
    solve: Solve = Solve()
