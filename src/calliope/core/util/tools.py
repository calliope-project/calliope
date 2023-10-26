# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).


from pathlib import Path
from typing import TypeVar

import jsonschema
from typing_extensions import ParamSpec

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
    jsonschema.Draft7Validator.META_SCHEMA["additionalProperties"] = False
    validator = jsonschema.Draft7Validator(schema)
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
        for error in sorted(validator.iter_errors(to_validate), key=str):
            path_ = error.json_path.lstrip("$.")
            message = error.args[0]
            if path_ == "":
                errors.append(message)
            else:
                errors.append(f"{path_}: {message}")
    if errors:
        print_warnings_and_raise_errors(
            errors=errors, during=f"validation of the {dict_descriptor} dictionary."
        )
