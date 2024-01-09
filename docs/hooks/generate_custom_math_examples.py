# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
generate_custom_math_examples.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates a markdown file listing all custom math examples.

"""
import tempfile
from io import StringIO
from pathlib import Path

import ruamel.yaml
from mkdocs.structure.files import File

TEMPDIR = tempfile.TemporaryDirectory()

CUSTOM_MATH_PATH = Path("docs") / "custom_math" / "examples"

MD_EXAMPLE_STRING = """
# {title}

## Description

{description}

## YAML definition

[:fontawesome-solid-download: Download the YAML example]({yaml_file_path})

```yaml
{example_yaml}
```
"""


def on_files(files: list, config: dict, **kwargs):
    """Generate custom math example markdown files and attach them to the documentation and the navigation tree."""

    # Find the navigation tree list that we will populate with reference to new markdown files
    top_level_nav_reference = [
        idx for idx in config["nav"] if set(idx.keys()) == {"Custom math"}
    ][0]
    nav_reference = [
        idx
        for idx in top_level_nav_reference["Custom math"]
        if isinstance(idx, dict) and set(idx.keys()) == {"Example custom math gallery"}
    ][0]

    # Generate and attach the markdown files
    for file in CUSTOM_MATH_PATH.glob("*.yaml"):
        files.append(
            _process_file(file, config, nav_reference["Example custom math gallery"])
        )
    return files


def _process_file(path: Path, config: dict, nav_reference: list) -> File:
    output_file = path.relative_to("docs").with_suffix(".md")
    output_full_filepath = Path(TEMPDIR.name) / output_file
    output_full_filepath.parent.mkdir(exist_ok=True, parents=True)

    text = path.read_text()

    comment_, yaml_ = text.split("# ---", 1)

    comment_yaml_block = "\n".join(i.removeprefix("#") for i in comment_.split("\n"))

    yaml = ruamel.yaml.YAML(typ="safe")
    comment_yaml = yaml.load(StringIO(comment_yaml_block))

    title = comment_yaml.get("title", "")
    description = comment_yaml.get("description", "")

    example_yaml = "\n".join(i for i in yaml_.split("\n"))

    formatted_string = MD_EXAMPLE_STRING.format(
        title=title,
        description=description,
        yaml_file_path=Path("..") / path.relative_to(CUSTOM_MATH_PATH),
        example_yaml=example_yaml,
    )
    output_full_filepath.write_text(formatted_string)
    nav_reference.append(output_file.as_posix())

    return File(
        path=output_file,
        src_dir=TEMPDIR.name,
        dest_dir=config["site_dir"],
        use_directory_urls=config["use_directory_urls"],
    )
