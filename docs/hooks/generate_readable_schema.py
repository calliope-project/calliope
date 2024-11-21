# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""Schema (a.k.a. "properties") documentation functionality.

Converts YAML schema to a readable markdown file with nested lists representing the
schema properties.
"""

import tempfile
import textwrap
from pathlib import Path

import jsonschema2md
from mkdocs.structure.files import File

from calliope.schemas import config_schema
from calliope.util import schema

TEMPDIR = tempfile.TemporaryDirectory()

SCHEMAS = {
    "config_schema": config_schema.CalliopeConfig().model_no_ref_schema(),
    "model_schema": schema.MODEL_SCHEMA,
    "math_schema": schema.MATH_SCHEMA,
    "data_table_schema": schema.DATA_TABLE_SCHEMA,
}


def on_files(files: list, config: dict, **kwargs):
    """Generate schema markdown reference sheets and attach them to the documentation."""
    for schema_name, schema_dict in SCHEMAS.items():
        files.append(_schema_to_md(schema_dict, schema_name, config))
    return files


def _schema_to_md(schema: dict, filename: str, config: dict) -> File:
    """Parse the schema to markdown, edit some lines, then dump to file.

    Args:
        schema (dict): Path to the YAML schema to be converted.
        filename (str): Path to Markdownfile to dump the converted schema.
        config (dict): Documentation configuration.

    Returns:
        File: markdown (.md) file.
    """
    output_file = (Path("reference") / filename).with_suffix(".md")
    output_full_filepath = Path(TEMPDIR.name) / output_file
    output_full_filepath.parent.mkdir(exist_ok=True, parents=True)

    parser = jsonschema2md.Parser()
    parser.tab_size = 4

    # We don't want to represent the schema as a schema,
    # so we remove parts of the generated markdown that refers to it as such
    lines = parser.parse_schema(schema)
    assert lines[2] == "## Properties\n\n"
    del lines[2]

    initial_lines = textwrap.dedent(
        """
        ---
        search:
          boost: 0.25
        ---

        """
    )
    output_full_filepath.write_text(initial_lines.lstrip() + "\n".join(lines))

    return File(
        path=output_file,
        src_dir=TEMPDIR.name,
        dest_dir=config["site_dir"],
        use_directory_urls=config["use_directory_urls"],
    )
