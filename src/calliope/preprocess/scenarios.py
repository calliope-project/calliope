# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Preprocessing of base model definition and overrides/scenarios into a unified dictionary."""

import logging

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.util.tools import listify

LOGGER = logging.getLogger(__name__)


def load_scenario_overrides(
    model_definition: dict,
    scenario: str | None = None,
    override_dict: dict | None = None,
    **kwargs,
) -> tuple[AttrDict, str]:
    """Apply user-defined overrides to the model definition.

    Args:
        model_definition (dict):
            Model definition dictionary.
        scenario (str | None, optional): Scenario(s) to apply, comma separated.
            e.g.: 'my_scenario_name' or 'override1,override2'.
            Defaults to None.
        override_dict (dict | None, optional):
            Overrides to apply _after_ `scenario` overrides.
            Defaults to None.
        **kwargs:
            initialisation overrides.

    Returns:
        tuple[AttrDict, str]:
            1. Model definition with overrides applied.
            2. Expansion of scenarios (which are references to model overrides) into a list of named override(s) that have been applied.
    """
    model_def_dict = AttrDict(model_definition)

    # The input files are allowed to override other model defaults
    model_def_with_overrides = model_def_dict.copy()

    # First pass of applying override dict before applying scenarios,
    # so that can override scenario definitions by override_dict
    if isinstance(override_dict, str):
        override_dict = AttrDict.from_yaml_string(override_dict)

    if isinstance(override_dict, dict):
        override_dict = AttrDict(override_dict)
        model_def_with_overrides.union(
            override_dict, allow_override=True, allow_replacement=True
        )

    overrides = model_def_with_overrides.pop("overrides", {})
    scenarios = model_def_with_overrides.pop("scenarios", {})

    if scenario is not None:
        applied_overrides = _load_overrides_from_scenario(
            model_def_with_overrides, scenario, overrides, scenarios
        )
        LOGGER.info(
            f"(scenarios, {scenario} ) | Applying the following overrides: {applied_overrides}."
        )
        overrides_from_scenario = _combine_overrides(overrides, applied_overrides)

        model_def_with_overrides.union(
            overrides_from_scenario, allow_override=True, allow_replacement=True
        )
    else:
        applied_overrides = []

    # Second pass of applying override dict after applying scenarios,
    # so that scenario-based overrides are overridden by override_dict!
    if override_dict is not None:
        model_def_with_overrides.union(
            override_dict, allow_override=True, allow_replacement=True
        )
    if "locations" in model_def_with_overrides.keys():
        # TODO: remove in v0.7.1
        exceptions.warn(
            "`locations` has been renamed to `nodes` and will stop working "
            "in v0.7.1. Please update your model configuration accordingly.",
            FutureWarning,
        )
        model_def_with_overrides["nodes"] = model_def_with_overrides["locations"]
        del model_def_with_overrides["locations"]

    _log_overrides(model_def_dict, model_def_with_overrides)

    model_def_with_overrides.union(
        AttrDict({"config.init": kwargs}), allow_override=True
    )

    return (model_def_with_overrides, ";".join(applied_overrides))


def _combine_overrides(overrides: AttrDict, scenario_overrides: list):
    combined_override_dict = AttrDict()
    for override in scenario_overrides:
        try:
            yaml_string = overrides[override].to_yaml()
            override_with_imports = AttrDict.from_yaml_string(yaml_string)
        except KeyError:
            raise exceptions.ModelError(f"Override `{override}` is not defined.")
        try:
            combined_override_dict.union(override_with_imports, allow_override=False)
        except KeyError as e:
            raise exceptions.ModelError(
                f"{str(e)[1:-1]}. Already specified but defined again in override `{override}`."
            )

    return combined_override_dict


def _load_overrides_from_scenario(
    model_def: AttrDict, scenario: str, overrides: dict, scenarios: dict
) -> list[str]:
    if scenario in scenarios.keys():
        LOGGER.info(f"Loading overrides from scenario: {scenario} ")
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
