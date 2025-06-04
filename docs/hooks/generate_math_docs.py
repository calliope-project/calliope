# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Generate LaTeX math to include in the documentation."""

import importlib.resources
import logging
import tempfile
import textwrap
from pathlib import Path

from mkdocs.structure.files import File

import calliope
from calliope.io import read_rich_yaml
from calliope.postprocess.math_documentation import MathDocumentation

logger = logging.getLogger("mkdocs")

TEMPDIR = tempfile.TemporaryDirectory()

MODEL_PATH = Path(__file__).parent / "dummy_model" / "model.yaml"

PREPEND_SNIPPET = """
# {title}
{description}

## A guide to math documentation

If a math component's initial conditions are met (the first `if` statement), it will be applied to a model.
For each [objective](#objective), [constraint](#subject-to) and [global expression](#where), a number of sub-conditions then apply (the subsequent, indented `if` statements) to decide on the specific expression to apply at a given iteration of the component dimensions.

In the expressions, terms in **bold** font are [decision variables](#decision-variables) and terms in *italic* font are [parameters](#parameters).
The [decision variables](#decision-variables) and [parameters](#parameters) are listed at the end of the page; they also refer back to the global expressions / constraints in which they are used.
Those parameters which are defined over time (`timesteps`) in the expressions can be defined by a user as a single, time invariant value, or as a timeseries that is [loaded from file or dataframe](../creating/data_tables.md).

!!! note

    For every math component in the documentation, we include the YAML snippet that was used to generate the math in a separate tab.

[:fontawesome-solid-download: Download the {math_type} formulation as a YAML file]({filepath})
"""


def on_files(files: list, config: dict, **kwargs):
    """Process documentation for pre-defined calliope math files."""
    model_config = read_rich_yaml(MODEL_PATH)

    base_documentation = generate_base_math_documentation()
    write_file(
        "plan.yaml",
        textwrap.dedent(
            """
        Complete base mathematical formulation for a Calliope model.
        This math is _always_ applied but can be overridden with pre-defined additional math or [your own math][adding-your-own-math-to-a-model].
        """
        ),
        base_documentation,
        files,
        config,
    )

    for override in model_config["overrides"].keys():
        custom_documentation = generate_custom_math_documentation(
            base_documentation, override
        )
        write_file(
            f"{override}.yaml",
            textwrap.dedent(
                f"""
            Pre-defined additional math to apply {custom_documentation.name} __on top of__ the [base mathematical formulation][base-math].
            This math is _only_ applied if referenced in the `config.build.extra_math` list as `{override}`.
            """
            ),
            custom_documentation,
            files,
            config,
        )

    return files


def write_file(
    filename: str,
    description: str,
    math_documentation: MathDocumentation,
    files: list[File],
    config: dict,
) -> None:
    """Parse math files and produce markdown documentation.

    Args:
        filename (str): name of produced `.md` file.
        description (str): first paragraph after title.
        math_documentation (MathDocumentation): calliope math documentation.
        files (list[File]): math files to parse.
        config (dict): documentation configuration.
    """
    output_file = (Path("math") / filename).with_suffix(".md")
    output_full_filepath = Path(TEMPDIR.name) / output_file
    output_full_filepath.parent.mkdir(exist_ok=True, parents=True)

    files.append(
        File(
            path=output_file.as_posix(),
            src_dir=TEMPDIR.name,
            dest_dir=config["site_dir"],
            use_directory_urls=config["use_directory_urls"],
        )
    )

    # Append the source file to make it available for direct download
    files.append(
        File(
            path=(Path("math") / filename).as_posix(),
            src_dir=Path(importlib.resources.files("calliope")).as_posix(),
            dest_dir=config["site_dir"],
            use_directory_urls=config["use_directory_urls"],
        )
    )
    nav_reference = [
        idx
        for idx in config["nav"]
        if isinstance(idx, dict) and set(idx.keys()) == {"Pre-defined math"}
    ][0]

    nav_reference["Pre-defined math"].append(output_file.as_posix())

    title = math_documentation.name
    math_doc = math_documentation.write(format="md", mkdocs_features=True)
    file_to_download = Path("..") / filename
    output_full_filepath.write_text(
        PREPEND_SNIPPET.format(
            title=title.capitalize(),
            description=description,
            math_type=title.lower(),
            filepath=file_to_download,
        )
        + math_doc
    )


def generate_base_math_documentation() -> MathDocumentation:
    """Generate model documentation for the base math.

    Returns:
        MathDocumentation: model math documentation with latex backend.
    """
    model = calliope.Model(model_definition=MODEL_PATH)
    model.build()
    return MathDocumentation(model)


def generate_custom_math_documentation(
    base_documentation: MathDocumentation, override: str
) -> MathDocumentation:
    """Generate model documentation for a pre-defined math file.

    Only the changes made relative to the base math will be shown.

    Args:
        base_documentation (MathDocumentation): model documentation with only the base math applied.
        override (str): Name of override to load from the list available in the model config.

    Returns:
        MathDocumentation: model math documentation with latex backend.
    """
    model = calliope.Model(model_definition=MODEL_PATH, scenario=override)
    model.build()

    full_del = []
    expr_del = []
    for component_group, component_group_dict in model.applied_math.items():
        for name, component_dict in component_group_dict.items():
            if name in base_documentation.math[component_group]:
                if not component_dict.get("active", True):
                    expr_del.append(name)
                    component_dict["description"] = "|REMOVED|"
                    component_dict["active"] = True
                elif (
                    base_documentation.math[component_group].get(name, {})
                    != component_dict
                ):
                    _add_to_description(component_dict, "|UPDATED|")
                else:
                    full_del.append(name)
            else:
                _add_to_description(component_dict, "|NEW|")

    math_documentation = MathDocumentation(model)
    for key in expr_del:
        math_documentation.backend._dataset[key].attrs["math_string"] = ""
    for key in full_del:
        del math_documentation.backend._dataset[key]
    for var in math_documentation.backend._dataset.values():
        var.attrs["references"] = var.attrs["references"].intersection(
            math_documentation.backend._dataset.keys()
        )
        var.attrs["references"] = var.attrs["references"].difference(expr_del)

    logger.info(math_documentation.backend._dataset["carrier_in"].attrs["references"])

    return math_documentation


def _add_to_description(component_dict: dict, update_string: str) -> None:
    """Prepend the math component description."""
    component_dict["description"] = f"{update_string}\n{component_dict['description']}"
