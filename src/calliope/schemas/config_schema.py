# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope configuration definition."""

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field

from calliope.schemas.general import CalliopeBaseModel, UniqueList

Mode = Literal["plan", "operate", "spores"]

LOGGER = logging.getLogger(__name__)

SPORES_SCORING_OPTIONS = Literal[
    "integer", "relative_deployment", "random", "evolving_average"
]


class Init(CalliopeBaseModel):
    """All configuration options used when initialising a Calliope model."""

    model_config = {"title": "Model initialisation configuration"}
    name: str | None = None
    """Model name"""

    calliope_version: str | None = None
    """Calliope framework version this model is intended for"""

    broadcast_param_data: bool = Field(default=False)
    """
    If True, single data entries in YAML indexed parameters will be broadcast across all index items.
    Otherwise, the number of data entries needs to match the number of index items.
    Defaults to False to mitigate unexpected broadcasting when applying overrides.
    """

    time_subset: tuple[str, str] | None = None
    """
    Subset of timesteps as an two-element list giving the **inclusive** range.
    For example, ["2005-01", "2005-04"] will create a time subset from "2005-01-01 00:00:00" to "2005-04-31 23:59:59".

    Strings must be ISO8601-compatible, i.e. of the form `YYYY-mm-dd HH:MM:SS` (e.g, '2005-01 ', '2005-01-01', '2005-01-01 00:00', ...)
    """

    time_resample: str | None = Field(default=None, pattern="^[0-9]+[a-zA-Z]")
    """Setting to adjust time resolution, e.g. '2h' for 2-hourly"""

    time_cluster: str | None = None
    """
    Setting to cluster the timeseries.
    Must be a path to a file where each date is linked to a representative date that also exists in the timeseries.
    """

    time_format: str = Field(default="ISO8601")
    """
    Timestamp format of all time series data when read from file.
    'ISO8601' means '%Y-%m-%d %H:%M:%S'.
    """

    distance_unit: Literal["km", "m"] = Field(default="km")
    """
    Unit of transmission link `distance` (m - metres, km - kilometres).
    Automatically derived distances from lat/lon coordinates will be given in this unit.
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

    use_cap_results: bool = Field(default=False)
    """If the model already contains `plan` mode results, use those optimal capacities as input parameters to the `operate` mode run."""


class Build(CalliopeBaseModel):
    """Base configuration options used when building a Calliope optimisation problem (`calliope.Model.build`)."""

    model_config = {"title": "Model build configuration"}
    mode: Mode = Field(default="plan")
    """Mode in which to run the optimisation."""

    add_math: UniqueList[str] = Field(default=[])
    """
    List of references to files which contain additional mathematical formulations to be applied on top of or instead of the base mode math.
    If referring to an pre-defined Calliope math file (see documentation for available files), do not append the reference with ".yaml".
    If referring to your own math file, ensure the file type is given as a suffix (".yaml" or ".yml").
    Relative paths will be assumed to be relative to the model definition file given when creating a calliope Model (`calliope.Model(model_definition=...)`)
    """

    ignore_mode_math: bool = Field(default=False)
    """
    If True, do not initialise the mathematical formulation with the pre-defined math for the given run `mode`.
    This option can be used to completely re-define the Calliope mathematical formulation.
    """

    backend: Literal["pyomo", "gurobi"] = Field(default="pyomo")
    """Module with which to build the optimisation problem."""

    ensure_feasibility: bool = Field(default=False)
    """
    Whether to include decision variables in the model which will meet unmet demand or consume unused supply in the model so that the optimisation solves successfully.
    This should only be used as a debugging option (as any unmet demand/unused supply is a sign of improper model formulation).
    """

    objective: str = Field(default="min_cost_optimisation")
    """Name of internal objective function to use, from those defined in the pre-defined math and any applied additional math."""

    pre_validate_math_strings: bool = Field(default=True)
    """
    If true, the Calliope math definition will be scanned for parsing errors _before_ undertaking the much more expensive operation of building the optimisation problem.
    You can switch this off (e.g., if you know there are no parsing errors) to reduce overall build time.
    """

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

    skip_baseline_run: bool = Field(default=False)
    """If the model already contains `plan` mode results, use those as the initial base run results and start with SPORES iterations immediately."""

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
    save_logs: Path | None = None
    """If given, should be a path to a directory in which to save optimisation logs."""

    solver_io: str | None = None
    """
    Some solvers have different interfaces that perform differently.
    For instance, setting `solver_io="python"` when using the solver `gurobi` tends to reduce the time to send the optimisation problem to the solver.
    """

    solver_options: dict = Field(default={})
    """Any solver options, as key-value pairs, to pass to the chosen solver"""

    solver: str = Field(default="cbc")
    """Solver to use. Any solvers that have Pyomo interfaces can be used. Refer to the Pyomo documentation for the latest list."""

    zero_threshold: float = Field(default=1e-10)
    """On postprocessing the optimisation results, values smaller than this threshold will be considered as optimisation artefacts and will be set to zero."""

    shadow_prices: UniqueList[str] = Field(default=[])
    """Names of model constraints."""

    spores: SolveSpores = SolveSpores()


class CalliopeConfig(CalliopeBaseModel):
    """Calliope configuration class."""

    model_config = {"title": "Model configuration schema"}

    init: Init = Init()
    build: Build = Build()
    solve: Solve = Solve()
