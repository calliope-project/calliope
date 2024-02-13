# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""
Convert plaintext example notebooks to .ipynb format and store them as `notebook.ipynb`
in every example notebook directory in the docs.
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
        if any(part.startswith(".") for part in file.parts):
            continue
        output_file = file.relative_to("docs").parent / file.stem / "notebook.ipynb"
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
