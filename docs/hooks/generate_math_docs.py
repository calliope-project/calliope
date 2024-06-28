# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Generate LaTeX math to include in the documentation."""

import importlib.resources
import logging
import tempfile
import textwrap
from pathlib import Path

import calliope
from mkdocs.structure.files import File

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
Those parameters which are defined over time (`timesteps`) in the expressions can be defined by a user as a single, time invariant value, or as a timeseries that is [loaded from file or dataframe](../creating/data_sources.md).

!!! note

    For every math component in the documentation, we include the YAML snippet that was used to generate the math in a separate tab.

[:fontawesome-solid-download: Download the {math_type} formulation as a YAML file]({filepath})
"""


def on_files(files: list, config: dict, **kwargs):
    """Process documentation for pre-defined calliope math files."""
    model_config = calliope.AttrDict.from_yaml(MODEL_PATH)

    base_model = generate_base_math_model()
    write_file(
        "base.yaml",
        textwrap.dedent(
            """
        Complete base mathematical formulation for a Calliope model.
        This math is _always_ applied but can be overridden with pre-defined additional math or [your own math][adding-your-own-math-to-a-model].
        """
        ),
        base_model,
        files,
        config,
    )

    for override in model_config["overrides"].keys():
        custom_model = generate_custom_math_model(base_model, override)
        write_file(
            f"{override}.yaml",
            textwrap.dedent(
                f"""
            Pre-defined additional math to apply {custom_model.inputs.attrs['name']} math on top of the [base mathematical formulation][base-math].
            This math is _only_ applied if referenced in the `config.init.add_math` list as `{override}`.
            """
            ),
            custom_model,
            files,
            config,
        )

    return files


def write_file(
    filename: str,
    description: str,
    model: calliope.Model,
    files: list[File],
    config: dict,
) -> None:
    """Parse math files and produce markdown documentation.

    Args:
        filename (str): name of produced `.md` file.
        description (str): first paragraph after title.
        model (calliope.Model): calliope model with the given math.
        files (list[File]): math files to parse.
        config (dict): documentation configuration.
    """
    title = model.inputs.attrs["name"] + " math"

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

    math_doc = model.math_documentation.write(format="md", mkdocs_tabbed=True)
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


def generate_base_math_model() -> calliope.Model:
    """Generate model with documentation for the base math.

    Args:
        model_config (dict): Calliope model config.

    Returns:
        calliope.Model: Base math model to use in generating math docs.
    """
    model = calliope.Model(model_definition=MODEL_PATH)
    model.math_documentation.build()
    return model


def generate_custom_math_model(
    base_model: calliope.Model, override: str
) -> calliope.Model:
    """Generate model with documentation for a pre-defined math file.

    Only the changes made relative to the base math will be shown.

    Args:
        base_model (calliope.Model): Calliope model with only the base math applied.
        override (str): Name of override to load from the list available in the model config.
    """
    model = calliope.Model(model_definition=MODEL_PATH, scenario=override)

    full_del = []
    expr_del = []
    for component_group, component_group_dict in model.math.items():
        for name, component_dict in component_group_dict.items():
            if name in base_model.math[component_group]:
                if not component_dict.get("active", True):
                    expr_del.append(name)
                    component_dict["description"] = "|REMOVED|"
                    component_dict["active"] = True
                elif base_model.math[component_group].get(name, {}) != component_dict:
                    _add_to_description(component_dict, "|UPDATED|")
                else:
                    full_del.append(name)
            else:
                _add_to_description(component_dict, "|NEW|")

    model.math_documentation.build()
    for key in expr_del:
        model.math_documentation._instance._dataset[key].attrs["math_string"] = ""
    for key in full_del:
        del model.math_documentation._instance._dataset[key]
    for var in model.math_documentation._instance._dataset.values():
        var.attrs["references"] = var.attrs["references"].intersection(
            model.math_documentation._instance._dataset.keys()
        )
        var.attrs["references"] = var.attrs["references"].difference(expr_del)

    logger.info(
        model.math_documentation._instance._dataset["carrier_in"].attrs["references"]
    )

    return model


def _add_to_description(component_dict: dict, update_string: str) -> None:
    """Prepend the math component description."""
    component_dict["description"] = f"{update_string}\n{component_dict['description']}"
