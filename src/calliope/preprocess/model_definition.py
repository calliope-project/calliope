# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Preprocessing of model definition into a unified dictionary."""

import logging
from pathlib import Path

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.io import read_rich_yaml, to_yaml
from calliope.util.tools import listify

LOGGER = logging.getLogger(__name__)


def prepare_model_definition(
    data: str | Path | dict,
    scenario: str | None = None,
    override_dict: dict | None = None,
) -> tuple[AttrDict, str]:
    """Arrange model definition data folloging our standardised order of priority.

    Should always be called when defining calliope models from configuration files.
    The order of priority is:

    - override_dict > scenarios > data section > template

    Args:
        data (str | Path | dict): model data file or dictionary.
        scenario (str | None, optional): scenario to run. Defaults to None.
        override_dict (dict | None, optional): additional overrides. Defaults to None.

    Returns:
        tuple[AttrDict, str]: _description_
    """
    if isinstance(data, dict):
        model_def = AttrDict(data)
    else:
        model_def = read_rich_yaml(data)
    model_def, applied_overrides = _load_scenario_overrides(
        model_def, scenario, override_dict
    )
    template_solver = TemplateSolver(model_def)
    model_def = template_solver.resolved_data

    return model_def, applied_overrides


def _load_scenario_overrides(
    model_definition: dict,
    scenario: str | None = None,
    override_dict: dict | None = None,
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

    Returns:
        tuple[AttrDict, str]:
            1. Model definition with overrides applied.
            2. Expansion of scenarios (which are references to model overrides) into a list of named override(s) that have been applied.
    """
    model_def_dict = AttrDict(model_definition)

    # The input files are allowed to override other model defaults
    model_def_with_overrides = model_def_dict.copy()

    # Apply override_dict first so it can overwrite scenario definitions
    overrides = AttrDict(override_dict)
    model_def_with_overrides.union(
        overrides, allow_override=True, allow_replacement=True
    )

    # Apply scenario overrides (with override_dict modifications if present)
    scenario_overrides = model_def_with_overrides.pop("overrides", {})
    scenarios = model_def_with_overrides.pop("scenarios", {})
    if scenario is not None:
        applied_overrides = _load_overrides_from_scenario(
            model_def_with_overrides, scenario, scenario_overrides, scenarios
        )
        LOGGER.info(
            f"(scenarios, {scenario} ) | Applying the following overrides: {applied_overrides}."
        )
        overrides_from_scenario = _combine_overrides(
            scenario_overrides, applied_overrides
        )
        model_def_with_overrides.union(
            overrides_from_scenario, allow_override=True, allow_replacement=True
        )
    else:
        applied_overrides = []

    # Second pass of applying override_dict after applying scenarios,
    # so it can overwrite scenarios (it has the highest priority)
    if overrides:
        # Remove scenarios/overrides since they were already applied.
        overrides.pop("scenarios", None)
        overrides.pop("overrides", None)
        model_def_with_overrides.union(
            overrides, allow_override=True, allow_replacement=True
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

    return (model_def_with_overrides, ";".join(applied_overrides))


def _combine_overrides(overrides: AttrDict, scenario_overrides: list[str]):
    combined_override_dict = AttrDict()
    for override in scenario_overrides:
        try:
            yaml_string = to_yaml(overrides[override])
            override_with_imports = read_rich_yaml(yaml_string)
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


class TemplateSolver:
    """Resolves templates before they reach Calliope models."""

    TEMPLATES_SECTION: str = "templates"
    TEMPLATE_CALL: str = "template"

    def __init__(self, data: AttrDict):
        """Initialise the solver."""
        self._raw_templates: AttrDict = data.get_key(self.TEMPLATES_SECTION, AttrDict())
        self._raw_data: AttrDict = data
        self.resolved_templates: AttrDict
        self.resolved_data: AttrDict
        self._resolve()

    def _resolve(self):
        """Fill in template references and remove template definitions and calls."""
        self.resolved_templates = AttrDict()
        for key, value in self._raw_templates.items():
            if not isinstance(value, dict):
                raise exceptions.ModelError("Template definitions must be YAML blocks.")
            self.resolved_templates[key] = self._resolve_template(key)
        self.resolved_data = self._resolve_data(self._raw_data)

    def _resolve_template(self, name: str, stack: None | set[str] = None) -> AttrDict:
        """Resolves templates recursively.

        Catches circular template definitions.
        """
        if stack is None:
            stack = set()
        elif name in stack:
            raise exceptions.ModelError(
                f"Circular template reference detected for '{name}'."
            )
        stack.add(name)

        result = AttrDict()
        raw_data = self._raw_templates[name]
        if self.TEMPLATE_CALL in raw_data:
            # Current template takes precedence when overriding values
            inherited_name = raw_data[self.TEMPLATE_CALL]
            if inherited_name in self.resolved_templates:
                inherited_data = self.resolved_templates[inherited_name]
            else:
                inherited_data = self._resolve_template(inherited_name, stack)
            result.union(inherited_data)

        local_data = {k: raw_data[k] for k in raw_data.keys() - {self.TEMPLATE_CALL}}
        result.union(local_data, allow_override=True)

        stack.remove(name)
        return result

    def _resolve_data(self, section, level: int = 0):
        if isinstance(section, dict):
            if self.TEMPLATES_SECTION in section:
                if level != 0:
                    raise exceptions.ModelError(
                        "Template definitions must be placed at the top level of the YAML file."
                    )
            if self.TEMPLATE_CALL in section:
                template = self.resolved_templates[section[self.TEMPLATE_CALL]].copy()
            else:
                template = AttrDict()

            local = AttrDict()
            for key in section.keys():
                if key not in [self.TEMPLATE_CALL, self.TEMPLATES_SECTION]:
                    local[key] = self._resolve_data(section[key], level=level + 1)

            # Local values have priority.
            template.union(local, allow_override=True)
            result = template
        else:
            result = section
        return result
