# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).


import re
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import jsonschema
from typing_extensions import ParamSpec

from calliope.attrdict import AttrDict
from calliope.exceptions import print_warnings_and_raise_errors

P = ParamSpec("P")
T = TypeVar("T")


def relative_path(base_path_file, path) -> Path:
    """
    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.

    """
    # Check if base_path_file is a string because it might be an AttrDict
    path = Path(path)
    if base_path_file is not None:
        base_path_file = Path(base_path_file)
        if base_path_file.is_file():
            base_path_file = base_path_file.parent
        if not path.is_absolute():
            path = base_path_file.absolute() / path
    return path


def update_then_validate_config(
    config_key: str, config_dict: AttrDict, schema: dict, **update_kwargs
) -> AttrDict:
    to_validate = deepcopy(config_dict[config_key])
    to_validate.union(AttrDict(update_kwargs), allow_override=True)
    validate_dict({config_key: to_validate}, schema, f"`{config_key}` configuration")
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


def extract_from_schema(schema: dict, keyword: str) -> dict:
    extracted_keywords: dict = {}
    KeywordValidatingValidator = _extend_with_keyword(
        jsonschema.Draft202012Validator, keyword
    )
    KeywordValidatingValidator(schema).validate(extracted_keywords)
    return extracted_keywords


def _extend_with_keyword(
    validator_class: jsonschema.Validator, keyword: str
) -> jsonschema.Validator:
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, val in properties.as_dict_flat().items():
            if prop.endswith(keyword):
                config_key_regex = rf"(init|build|solve)\.properties\.(\w+)\.{keyword}"
                model_def_key_regex = rf".*\.properties\.(\w+)\.{keyword}"
                new_key = re.match(config_key_regex, prop)
                if new_key is None:
                    new_key = re.match(model_def_key_regex, prop)
                instance.setdefault(".".join(new_key.groups()), val)

        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    return jsonschema.validators.extend(
        validator_class,
        {"properties": set_defaults},
    )


def listify(var: Any) -> list:
    """Transform the input into a list.

    If it is a non-string iterable, it will be coerced to a list.
    If it is a string or non-iterable (numeric, boolean, ...), then it will be a single item list.

    Args:
        var (Any): Value to transform.

    Returns:
        list: List containing `var` or elements of `var` (if input was a non-string iterable).
    """
    if not isinstance(var, str) and hasattr(var, "__iter__"):
        var = list(var)
    else:
        var = [var]
    return var
