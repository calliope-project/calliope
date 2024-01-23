# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
model_run.py
~~~~~~~~~~~~

Preprocessing of base model definition and overrides/scenarios into a unified dictionary

"""

import logging
from pathlib import Path
from typing import Optional

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.util.tools import listify

LOGGER = logging.getLogger(__name__)


def load_model_definition(
    model_definition: str | Path | dict,
    scenario: Optional[str] = None,
    override_dict: Optional[dict] = None,
    **kwargs,
) -> tuple[AttrDict, Optional[Path], str]:
    """
    Load model definition from file / dictionary and apply user-defined overrides.

    Args:
        model_definition (str | Path | dict):
            If string or pathlib.Path, path to YAML file with model configuration.
            If dictionary, equivalent to loading the model configuration YAML from file.
        scenario (Optional[str], optional):
            If not None, name of scenario to apply.
            Can either be a named scenario, or a comma-separated list of individual overrides to be combined ad-hoc,
            e.g. 'my_scenario_name' or 'override1,override2'.
            Defaults to None.
        override_dict (Optional[dict], optional):
            If not None, dictionary of overrides to apply.
            These will be applied _after_ `scenario` overrides.
            Defaults to None.

    Returns:
        tuple[AttrDict, Optional[Path], str]:
            1. Model definition with overrides applied.
            1. Path to model definition YAML if input `model_definiton` was pathlike, otherwise None.
            1. Expansion of scenarios (which are references to model overrides) into a list of named override(s) that have been applied.
    """
    if not isinstance(model_definition, dict):
        model_def_path = Path(model_definition)
        model_def_dict = AttrDict.from_yaml(model_def_path)
    else:
        model_def_dict = AttrDict(model_definition)
        model_def_path = None

    model_def_with_overrides, applied_overrides = _apply_overrides(
        model_def_dict, scenario=scenario, override_dict=override_dict
    )
    model_def_with_overrides.union(
        AttrDict({"config.init": kwargs}), allow_override=True
    )

    return (model_def_with_overrides, model_def_path, ";".join(applied_overrides))


def _combine_overrides(overrides: AttrDict, scenario_overrides: list):
    combined_override_dict = AttrDict()
    for override in scenario_overrides:
        try:
            yaml_string = overrides[override].to_yaml()
            override_with_imports = AttrDict.from_yaml_string(yaml_string)
        except KeyError:
            raise exceptions.ModelError(
                "Override `{}` is not defined.".format(override)
            )
        try:
            combined_override_dict.union(override_with_imports, allow_override=False)
        except KeyError as e:
            raise exceptions.ModelError(
                str(e)[1:-1] + ". Already specified but defined again in "
                "override `{}`.".format(override)
            )

    return combined_override_dict


def _apply_overrides(
    model_def: AttrDict,
    scenario: Optional[str] = None,
    override_dict: Optional[str | dict] = None,
) -> tuple[AttrDict, list[str]]:
    """
    Generate processed Model configuration, applying any scenario overrides.

    Args:
        model_def (calliope.Attrdict): Loaded model definition as an attribute dictionary.
        scenario (Optional[str], optional):
            If not None, name of scenario to apply.
            Can either be a named scenario, or a comma-separated list of individual overrides to be combined ad-hoc,
            e.g. 'my_scenario_name' or 'override1,override2'.
            Defaults to None.
        override_dict (Optional[dict], optional):
            If not None, dictionary of overrides to apply.
            These will be applied _after_ `scenario` overrides.
            Defaults to None.

    Returns:
        tuple[AttrDict, list[str]]:
            1. Model definition dictionary with overrides applied from `scenario` and `override_dict`.
            1. Expansion of scenarios (which are references to model overrides) into a list of named override(s) that have been applied.
    """

    # The input files are allowed to override other model defaults
    model_def_copy = model_def.copy()

    # First pass of applying override dict before applying scenarios,
    # so that can override scenario definitions by override_dict
    if isinstance(override_dict, str):
        override_dict = AttrDict.from_yaml_string(override_dict)

    if isinstance(override_dict, dict):
        override_dict = AttrDict(override_dict)
        model_def_copy.union(override_dict, allow_override=True, allow_replacement=True)

    overrides = model_def_copy.pop("overrides", {})
    scenarios = model_def_copy.pop("scenarios", {})

    if scenario is not None:
        scenario_overrides = _load_overrides_from_scenario(
            model_def_copy, scenario, overrides, scenarios
        )
        LOGGER.info(
            f"(scenarios, {scenario} ) | Applying the following overrides: {scenario_overrides}."
        )
        overrides_from_scenario = _combine_overrides(overrides, scenario_overrides)

        model_def_copy.union(
            overrides_from_scenario, allow_override=True, allow_replacement=True
        )
    else:
        scenario_overrides = []

    # Second pass of applying override dict after applying scenarios,
    # so that scenario-based overrides are overridden by override_dict!
    if override_dict is not None:
        model_def_copy.union(override_dict, allow_override=True, allow_replacement=True)
    if "locations" in model_def_copy.keys():
        # TODO: remove in v0.7.1
        exceptions.warn(
            "`locations` has been renamed to `nodes` and will stop working "
            "in v0.7.1. Please update your model configuration accordingly.",
            DeprecationWarning,
        )
        model_def_copy["nodes"] = model_def_copy["locations"]
        del model_def_copy["locations"]

    _log_overrides(model_def, model_def_copy)

    return model_def_copy, scenario_overrides


def _load_overrides_from_scenario(
    model_def: AttrDict, scenario: str, overrides: dict, scenarios: dict
) -> list[str]:
    if scenario in scenarios.keys():
        LOGGER.info("Loading overrides from scenario: {} ".format(scenario))
        scenario_list = listify(scenarios[scenario])
    else:
        scenario_list = scenario.split(",")
    scenario_overrides = set()
    for override in scenario_list:
        if isinstance(override, dict):
            raise exceptions.ModelError(
                "(scenarios, {scenario}) | Scenario definition must be a list of override or other scenario names."
            )
        if override in scenarios:
            scenario_overrides.update(
                _load_overrides_from_scenario(model_def, override, overrides, scenarios)
            )
        elif override not in overrides.keys():
            raise exceptions.ModelError(
                f"(scenarios, {scenario}) | Unrecognised override name: {override}."
            )
        else:
            scenario_overrides.add(override)

    return list(scenario_overrides)


def _log_overrides(init_model_def: AttrDict, overriden_model_def: AttrDict) -> None:
    init_model_def_flat = init_model_def.as_dict_flat()
    for key, val in overriden_model_def.as_dict_flat().items():
        if key in init_model_def_flat and init_model_def_flat[key] != val:
            message = (
                f"Override applied to `{key}`: {init_model_def_flat[key]} -> {val}"
            )
        elif key not in init_model_def_flat:
            message = f"New model config from override: `{key}`={val}"
        else:
            continue
        LOGGER.debug(message)
