import importlib
import re
from copy import deepcopy
from typing import Literal, Optional

import jsonschema

from calliope.attrdict import AttrDict
from calliope.exceptions import print_warnings_and_raise_errors

CONFIG_DIR = importlib.resources.files("calliope") / "config"
CONFIG_SCHEMA = AttrDict.from_yaml(CONFIG_DIR / "config_schema.yaml")
MODEL_SCHEMA = AttrDict.from_yaml(CONFIG_DIR / "model_def_schema.yaml")
MATH_SCHEMA = AttrDict.from_yaml(CONFIG_DIR / "math_schema.yaml")


def update_then_validate_config(
    config_key: str, config_dict: AttrDict, **update_kwargs
) -> AttrDict:
    to_validate = deepcopy(config_dict[config_key])
    to_validate.union(AttrDict(update_kwargs), allow_override=True)
    validate_dict(
        {"config": {config_key: to_validate}},
        CONFIG_SCHEMA,
        f"`{config_key}` configuration",
    )
    return to_validate


def validate_dict(to_validate: dict, schema: dict, dict_descriptor: str) -> None:
    """
    Validate a dictionary under a given schema.

    Args:
        to_validate (dict): Dictionary to validate.
        schema (dict): Schema to validate with.
        dict_descriptor (str): Description of the dictionary to validate, to use if an error is raised.

    Raises:
        jsonschema.SchemaError: If the schema itself is malformed, a SchemaError will be raised at the first issue. Other issues than that raised may still exist.
        calliope.exceptions.ModelError: If the dictionary is not valid according to the schema, a list of the issues found will be collated and raised.
    """
    errors = []
    validator = jsonschema.Draft202012Validator
    validator.META_SCHEMA["unevaluatedProperties"] = False
    try:
        validator.check_schema(schema)
    except jsonschema.SchemaError as err:
        path = ".".join(err.path)
        if path != "":
            path = f" at `{path}`"

        if err.context:
            message = err.context[0].args[0]
        else:
            message = err.args[0]
        raise jsonschema.SchemaError(
            message=f"The {dict_descriptor} schema is malformed{path}: {message}"
        )

    try:
        jsonschema.validate(to_validate, schema)
    except jsonschema.ValidationError:
        # TODO: update when OneOf errors are better formatted.
        # See https://github.com/python-jsonschema/jsonschema/issues/1002
        for err_ in sorted(validator(schema).iter_errors(to_validate), key=str):
            best_match = jsonschema.exceptions.best_match([err_])
            path_ = best_match.json_path.lstrip("$.")
            if path_ == "":
                errors.append(best_match.message)
            else:
                errors.append(f"{path_}: {best_match.message}")

    if errors:
        print_warnings_and_raise_errors(
            errors=errors, during=f"validation of the {dict_descriptor} dictionary."
        )


def extract_from_schema(
    schema: dict,
    keyword: str,
    subset_top_level: Optional[Literal["nodes", "techs", "parameters"]] = None,
) -> dict:
    """Extract a keyword for each leaf property in the schema.

    This currently only reliably works for "default".
    Other keywords exist at branch properties, which confuses the extraction process.

    Args:
        schema (dict): Schema to extract keyword from
        keyword (str): property key to extract
        subset_top_level (Literal["nodes", "techs", "parameters"], optional):
            Include only those properties that are leaves along a specific top-level property branch.
            Defaults to None, i.e., all property branches are included.

    Returns:
        dict: Flat dictionary of property name : keyword value.
        Property trees are discarded since property names must be unique.
    """
    extracted_keywords: dict = {}
    KeywordValidatingValidator = _extend_with_keyword(
        jsonschema.Draft202012Validator,
        keyword,
        subset_top_level if subset_top_level is not None else "",
    )
    KeywordValidatingValidator(schema).validate(extracted_keywords)
    return extracted_keywords


def _extend_with_keyword(
    validator_class: jsonschema.Validator, keyword: str, subset_top_level: str
) -> jsonschema.Validator:
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, val in properties.as_dict_flat().items():
            if prop.endswith(keyword):
                config_key_regex = rf"config\.properties\.(init|build|solve)\.properties\.(\w+)\.{keyword}"
                model_def_key_regex = (
                    rf"{subset_top_level}.*\.properties\.(\w+)\.{keyword}"
                )
                new_key = re.match(config_key_regex, prop)
                if new_key is None:
                    new_key = re.match(model_def_key_regex, prop)
                if new_key is None:
                    continue
                instance.setdefault(".".join(new_key.groups()), val)

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return jsonschema.validators.extend(validator_class, {"properties": set_defaults})
