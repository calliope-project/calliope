# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
generate_readable_schema.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts YAML schema to a readable markdown file with nested lists representing the schema properties.

"""

import tempfile
from pathlib import Path

import jsonschema2md
from calliope.util import schema
from mkdocs.structure.files import File

TEMPDIR = tempfile.TemporaryDirectory()

SCHEMAS = {
    "config_schema": schema.CONFIG_SCHEMA,
    "model_schema": schema.MODEL_SCHEMA,
    "math_schema": schema.MATH_SCHEMA,
}


def on_files(files: list, config: dict, **kwargs):
    """Generate schema markdown reference sheets and attach them to the documentation."""
    for schema_name, schema_dict in SCHEMAS.items():
        files.append(_schema_to_md(schema_dict, schema_name, config))
    return files


def _schema_to_md(schema: dict, filename: str, config: dict) -> File:
    """Parse the schema to markdown, edit some lines, then dump to file

    Args:
        schema (dict): Path to the YAML schema to be converted.
        path_to_md (Path): Path to Markdownfile to dump the converted schema.
    """
    output_file = (Path("reference") / filename).with_suffix(".md")
    output_full_filepath = Path(TEMPDIR.name) / output_file
    output_full_filepath.parent.mkdir(exist_ok=True, parents=True)

    parser = jsonschema2md.Parser()
    parser.tab_size = 2

    lines = parser.parse_schema(schema)
    lines = _customise_markdown(lines)

    output_full_filepath.write_text("\n".join(lines))

    return File(
        path=output_file,
        src_dir=TEMPDIR.name,
        dest_dir=config["site_dir"],
        use_directory_urls=config["use_directory_urls"],
    )


def _customise_markdown(lines: list[str]) -> list[str]:
    """
    We don't want to represent the schema as a schema, so we remove parts of the generated markdown that refers to it as such.
    """
    # 1. Remove subheadline
    assert lines[2] == "## Properties\n\n"
    del lines[2]
    return lines
