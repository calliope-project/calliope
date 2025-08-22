# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Load, update, and access attributes in the Calliope pre-defined YAML schemas."""

import jsonschema

from calliope.exceptions import print_warnings_and_raise_errors
from calliope.io import load_config

MODEL_SCHEMA = load_config("model_def_schema.yaml")


def validate_dict(to_validate: dict, schema: dict, dict_descriptor: str) -> None:
    """Validate a dictionary under a given schema.

    Args:
        to_validate (dict): Dictionary to validate.
        schema (dict): Schema to validate with.
        dict_descriptor (str): Description of the dictionary to validate, to use if an error is raised.

    Raises:
        jsonschema.SchemaError:
            If the schema itself is malformed, a SchemaError will be raised at the first issue.
            Other issues than that raised may still exist.
        calliope.exceptions.ModelError:
            If the dictionary is not valid according to the schema, a list of the issues found will be collated and raised.
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
