# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
generate_readable_schema.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts YAML schema to a readable markdown file with nested lists representing the schema properties.

"""

from pathlib import Path
from typing import Union

import jsonschema2md
import yaml


def schema_to_md(
    path_to_schema: Union[str, Path], path_to_md: Union[str, Path]
) -> None:
    """Parse the schema to markdown, edit some lines, then dump to file

    Args:
        path_to_schema (Union[str, Path]): Path to the YAML schema to be converted.
        path_to_md (Union[str, Path]): Path to Markdownfile to dump the converted schema.
    """
    parser = jsonschema2md.Parser()
    parser.tab_size = 2

    with Path(path_to_schema).open("r") as f_schema:
        schema = yaml.safe_load(f_schema)

    lines = parser.parse_schema(schema)
    lines = customise_markdown(lines)
    Path(path_to_md).write_text("\n".join(lines))


def customise_markdown(lines: list[str]) -> list[str]:
    """
    We don't want to represent the schema as a schema, so we remove parts of the generated markdown that refers to it as such.
    """
    # 1. Remove headline
    assert lines[0] == "# JSON Schema\n\n"
    del lines[0]
    # 2. Remove main description and subheadline
    assert lines[1] == "## Properties\n\n"
    del lines[1]
    return lines


def process():
    math_schema = Path("../calliope/config/math_schema.yaml")
    assert math_schema.is_file()
    schema_to_md(math_schema, "./user/includes/math_schema.md")
