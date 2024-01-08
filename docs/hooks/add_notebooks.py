# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
add_notebooks.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts YAML schema to a readable markdown file with nested lists representing the schema properties.

"""

import subprocess
import tempfile
from pathlib import Path

from mkdocs.structure.files import File

TEMPDIR = tempfile.TemporaryDirectory()

NOTEBOOK_DIR = Path("docs") / "examples"


def on_files(files: list, config: dict, **kwargs):
    """Generate schema markdown reference sheets and attach them to the documentation."""

    for file in NOTEBOOK_DIR.glob("**/*.py"):
        output_file = (
            file.relative_to("docs").parent
            / file.stem
            / file.with_suffix(".ipynb").name
        )
        output_temp_file = Path(TEMPDIR.name) / output_file
        output_temp_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"jupytext --to ipynb -o {output_temp_file} {file}", shell=True)
        file_obj = File(
            path=output_file,
            src_dir=TEMPDIR.name,
            dest_dir=config["site_dir"],
            use_directory_urls=config["use_directory_urls"],
        )
        files.append(file_obj)
    return files
