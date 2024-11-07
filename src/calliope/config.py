# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Implements the Calliope configuration class."""

from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar, get_args, overload

import jsonref
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError

from calliope.attrdict import AttrDict
from calliope.util import tools

MODES_T = Literal["plan", "operate", "spores"]
CONFIG_T = Literal["init", "build", "solve"]

# ==
# Taken from https://github.com/pydantic/pydantic-core/pull/820#issuecomment-1670475909
T = TypeVar("T", bound=Hashable)


def _validate_unique_list(v: list[T]) -> list[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


UniqueList = Annotated[
    list[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]
# ==


def hide_from_schema(to_hide: list[str]):
    """Hide fields from the generated schema.

    Args:
        to_hide (list[str]): List of fields to hide.
    """

    def _hide_from_schema(schema: dict):
        for hide in to_hide:
            schema.get("properties", {}).pop(hide, None)
        return schema

    return _hide_from_schema


class ConfigBaseModel(BaseModel):
    """A base class for creating pydantic models for Calliope configuration options."""

    _kwargs: dict = {}

    def update(self, update_dict: dict, deep: bool = False) -> Self:
        """Return a new iteration of the model with updated fields.

        Updates are validated and stored in the parent class in the `_kwargs` key.

        Args:
            update_dict (dict): Dictionary with which to update the base model.
            deep (bool, optional): Set to True to make a deep copy of the model. Defaults to False.

        Returns:
            BaseModel: New model instance.
        """
        new_dict: dict = {}
        # Iterate through dict to be updated and convert any sub-dicts into their respective pydantic model objects
        for key, val in update_dict.items():
            key_class = getattr(self, key)
            if isinstance(key_class, ConfigBaseModel):
                new_dict[key] = key_class.update(val)
                key_class._kwargs = val
            else:
                new_dict[key] = val
        updated = super().model_copy(update=new_dict, deep=deep)
        updated.model_validate(updated)
        self._kwargs = update_dict
        return updated

    @overload
    def model_yaml_schema(self, filepath: str | Path) -> None: ...

    @overload
    def model_yaml_schema(self, filepath: None = None) -> str: ...

    def model_yaml_schema(self, filepath: str | Path | None = None) -> None | str:
        """Generate a YAML schema for the class.

        Args:
            filepath (str | Path | None, optional): If given, save schema to given path. Defaults to None.

        Returns:
            None | str: If `filepath` is given, returns None. Otherwise, returns the YAML string.
        """
        # By default, the schema uses $ref/$def cross-referencing for each pydantic model class,
        # but this isn't very readable when rendered in our documentation.
        # So, we resolve references and then delete all the `$defs`
        schema_dict = AttrDict(jsonref.replace_refs(self.model_json_schema()))
        schema_dict.del_key("$defs")
        return schema_dict.to_yaml(filepath)

    @property
    def applied_keyword_overrides(self) -> dict:
        """Most recently applied keyword overrides used to update this configuration.

        Returns:
            dict: Description of applied overrides.
        """
        return self._kwargs


class ModeBaseModel(ConfigBaseModel):
    """Mode-specific configuration, which will be hidden from the string representation of the model if that mode is not activated."""

    mode: MODES_T = Field(default="plan")
    """Mode in which to run the optimisation."""

    @model_validator(mode="after")
    def update_repr(self) -> Self:
        """Hide config from model string representation if mode is not activated."""
        for key, val in self.model_fields.items():
            if key in get_args(MODES_T):
                val.repr = self.mode == key
        return self


class Init(ConfigBaseModel):
    """All configuration options used when initialising a Calliope model."""

    model_config = {
        "title": "init",
        "extra": "forbid",
        "frozen": True,
        "json_schema_extra": hide_from_schema(["def_path"]),
        "revalidate_instances": "always",
        "use_attribute_docstrings": True,
    }

    def_path: Path = Field(default=".", repr=False, exclude=True)
    """The path to the main model definition YAML file, if one has been used to instantiate the Calliope Model class."""

    name: str | None = Field(default=None)
    """Model name"""

    calliope_version: str | None = Field(default=None)
    """Calliope framework version this model is intended for"""

    time_subset: tuple[datetime, datetime] | None = Field(default=None)
    """
    Subset of timesteps as an two-element list giving the **inclusive** range.
    For example, ["2005-01", "2005-04"] will create a time subset from "2005-01-01 00:00:00" to "2005-04-31 23:59:59".

    Strings must be ISO8601-compatible, i.e. of the form `YYYY-mm-dd HH:MM:SS` (e.g, '2005-01 ', '2005-01-01', '2005-01-01 00:00', ...)
    """

    time_resample: str | None = Field(default=None, pattern="^[0-9]+[a-zA-Z]")
    """Setting to adjust time resolution, e.g. '2h' for 2-hourly"""

    time_cluster: Path | None = Field(default=None)
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

    @model_validator(mode="before")
    @classmethod
    def abs_path(cls, data):
        """Add model definition path."""
        if data.get("time_cluster", None) is not None:
            data["time_cluster"] = tools.relative_path(
                data["def_path"], data["time_cluster"]
            )
        return data


class BuildOperate(ConfigBaseModel):
    """Operate mode configuration options used when building a Calliope optimisation problem (`calliope.Model.build`)."""

    model_config = {
        "title": "operate",
        "extra": "forbid",
        "json_schema_extra": hide_from_schema(["start_window_idx"]),
        "revalidate_instances": "always",
        "use_attribute_docstrings": True,
    }

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

    start_window_idx: int = Field(default=0, repr=False, exclude=True)
    """Which time window to build. This is used to track the window when re-building the model part way through solving in `operate` mode."""


class Build(ModeBaseModel):
    """Base configuration options used when building a Calliope optimisation problem (`calliope.Model.build`)."""

    model_config = {
        "title": "build",
        "extra": "allow",
        "revalidate_instances": "always",
    }
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


class SolveSpores(ConfigBaseModel):
    """SPORES configuration options used when solving a Calliope optimisation problem (`calliope.Model.solve`)."""

    number: int = Field(default=3)
    """SPORES mode number of iterations after the initial base run."""

    score_cost_class: str = Field(default="score")
    """SPORES mode cost class to vary between iterations after the initial base run."""

    slack_cost_group: str = Field(default=None)
    """SPORES mode cost class to keep below the given `slack` (usually "monetary")."""

    save_per_spore: bool = Field(default=False)
    """
    Whether or not to save the result of each SPORES mode run between iterations.
    If False, will consolidate all iterations into one dataset after completion of N iterations (defined by `number`) and save that one dataset.
    """

    save_per_spore_path: Path | None = Field(default=None)
    """If saving per spore, the path to save to."""

    skip_cost_op: bool = Field(default=False)
    """If the model already contains `plan` mode results, use those as the initial base run results and start with SPORES iterations immediately."""

    @model_validator(mode="after")
    def require_save_per_spore_path(self) -> Self:
        """Ensure that path is given if saving per spore."""
        if self.save_per_spore:
            if self.save_per_spore_path is None:
                raise ValueError(
                    "Must define `save_per_spore_path` if you want to save each SPORES result separately."
                )
            elif not self.save_per_spore_path.is_dir():
                raise ValueError("`save_per_spore_path` must be a directory.")
        return self


class Solve(ModeBaseModel):
    """Base configuration options used when solving a Calliope optimisation problem (`calliope.Model.solve`)."""

    model_config = {
        "title": "solve",
        "extra": "forbid",
        "revalidate_instances": "always",
        "json_schema_extra": hide_from_schema(["mode"]),
    }

    save_logs: Path | None = Field(default=None)
    """If given, should be a path to a directory in which to save optimisation logs."""

    solver_io: str | None = Field(default=None)
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


class CalliopeConfig(ConfigBaseModel):
    """Calliope configuration class."""

    model_config = {"title": "config"}
    init: Init = Init()
    build: Build = Build()
    solve: Solve = Solve()

    @model_validator(mode="before")
    @classmethod
    def update_solve_mode(cls, data):
        """Solve mode should match build mode."""
        data["solve"]["mode"] = data["build"]["mode"]
        return data

    def update(self, update_dict: dict, deep: bool = False) -> Self:
        """Return a new iteration of the model with updated fields.

        Updates are validated and stored in the parent class in the `_kwargs` key.

        Args:
            update_dict (dict): Dictionary with which to update the base model.
            deep (bool, optional): Set to True to make a deep copy of the model. Defaults to False.

        Returns:
            BaseModel: New model instance.
        """
        update_dict_temp = AttrDict(update_dict)
        if update_dict_temp.get_key("build.mode", None) is not None:
            update_dict_temp.set_key("solve.mode", update_dict_temp["build"]["mode"])
        updated = super().update(update_dict_temp.as_dict(), deep=deep)
        return updated
