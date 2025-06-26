# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Command-line interface."""

import contextlib
import datetime
import itertools
import os
import pstats
import shutil
import sys
import traceback
from pathlib import Path

import click

from calliope import examples, io, read_netcdf, read_yaml
from calliope._version import __version__
from calliope.exceptions import BackendError
from calliope.util.generate_runs import generate
from calliope.util.logging import set_log_verbosity

_time_format = "%Y-%m-%d %H:%M:%S"


_debug = click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print debug information when encountering errors.",
)

_quiet = click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Be less verbose about what is happening, including hiding solver output.",
)

_pdb = click.option(
    "--pdb",
    is_flag=True,
    default=False,
    help="If used together with --debug, drop into interactive "
    "debugger on encountering errors.",
)

_profile = click.option(
    "--profile", is_flag=True, default=False, help="Run through cProfile."
)

_profile_filename = click.option(
    "--profile_filename",
    type=str,
    help="Filename to save profile to if enabled --profile.",
)


_fail_when_infeasible = click.option(
    "--fail_when_infeasible/--no_fail_when_infeasible",
    is_flag=True,
    default=True,
    help="Return fail on command line when problem is infeasible (default True).",
)


@contextlib.contextmanager
def format_exceptions(
    debug=False, pdb=False, profile=False, profile_filename=None, start_time=None
):
    """Exception formatting for better reading."""
    try:
        if profile:
            import cProfile

            profile = cProfile.Profile()
            profile.enable()
        yield
        if profile:
            profile.disable()
            if profile_filename:
                dump_path = os.path.expanduser(profile_filename)
                click.secho(f"\nSaving cProfile output to: {dump_path}")
                profile.dump_stats(dump_path)
            else:
                click.secho("\n\n----PROFILE OUTPUT----\n\n")
                stats = pstats.Stats(profile).sort_stats("cumulative")
                stats.print_stats(20)  # Print first 20 lines

    except Exception as e:
        if debug:
            traceback.print_exc()
            if pdb:
                import ipdb

                ipdb.post_mortem(e.__traceback__)
        else:
            stack = traceback.extract_tb(e.__traceback__)
            # Get last stack trace entry still in Calliope
            last = [i for i in stack if "calliope" in i[0]][-1]
            err_string = f"\nError in {last[2]}, {last[0]}:{last[1]}"
            click.secho(err_string, fg="red")
            click.secho(str(e), fg="red")
            if start_time:
                print_end_time(start_time, msg="aborted due to an error")
        sys.exit(1)


def print_end_time(start_time, msg="complete"):
    """Track calliope run time."""
    end_time = datetime.datetime.now()
    secs = round((end_time - start_time).total_seconds(), 1)
    tend = end_time.strftime(_time_format)
    click.secho(f"\nCalliope run {msg}. Elapsed: {secs} seconds (time at exit: {tend})")


def _get_version():
    return f"Version {__version__}"


def _cli_start(debug, quiet):
    """Initial setup for CLI commands.

    Returns ``start_time`` (datetime timestamp).
    """
    if debug:
        click.secho(_get_version())
        verbosity = "debug"
        log_solver = True
    else:
        if quiet:
            verbosity = "warning"
            log_solver = False
        else:  # Default option
            verbosity = "info"
            log_solver = True

    set_log_verbosity(
        verbosity, include_solver_output=log_solver, capture_warnings=True
    )

    start_time = datetime.datetime.now()

    return start_time


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", is_flag=True, default=False, help="Display version.")
def cli(ctx, version):
    """Calliope: a multi-scale energy systems modelling framework."""
    if ctx.invoked_subcommand is None and not version:
        click.secho(ctx.get_help())
    if version:
        click.secho(_get_version())


@cli.command(name="new", short_help="Create a new model based on a built-in example.")
@click.argument("path")
@click.option("--template", type=str, default=None)
@_debug
def new(path, template, debug):
    """Create new model at the given ``path`` based on one of the built-in example models.

    The target path must not yet exist.
    Intermediate directories will be created automatically.
    """
    _cli_start(debug, quiet=False)

    with format_exceptions(debug):
        if template is None:
            template = "national_scale"
        source_path = examples._EXAMPLE_MODEL_DIR / template
        click.echo(f"Copying {template} template to target directory: {path}")
        shutil.copytree(source_path, path)


def _run_setup_model(model_file, scenario, model_format, override_dict):
    """Build model in CLI commands.

    Returns ``model``, a ready-to-run calliope.Model instance.
    """
    # Try to determine model file type if not given explicitly
    if model_format is None:
        if model_file.split(".")[-1] in ["yaml", "yml"]:
            model_format = "yaml"
        elif model_file.split(".")[-1] in ["nc", "nc4", "netcdf"]:
            model_format = "netcdf"
        else:
            raise ValueError(
                "Cannot determine model file format based on file "
                f'extension for "{model_file}". Set format explicitly with '
                "--model_format."
            )
    if isinstance(override_dict, str):
        overrides = io.read_rich_yaml(override_dict)
    else:
        overrides = io.AttrDict()

    if model_format == "yaml":
        model = read_yaml(model_file, scenario=scenario, override_dict=overrides)
    elif model_format == "netcdf":
        if scenario is not None or override_dict is not None:
            raise ValueError(
                "When loading a pre-built model from NetCDF, the "
                "--scenario and --override_dict options are not available."
            )
        model = read_netcdf(model_file)
    else:
        raise ValueError(f"Invalid model format: {model_format}")

    return model


@cli.command(name="run", short_help="Build and solve a model.")
@click.argument("model_file")
@click.option("--scenario")
@click.option("--model_format")
@click.option("--override_dict")
@click.option("--save_netcdf")
@click.option("--save_csv")
@click.option("--save_logs")
@click.option(
    "--save_lp",
    help="Build and save model to the given LP file. "
    "When this is set, the model is not sent to a solver, and all other save options are ignored.",
)
@_debug
@_quiet
@_pdb
@_profile
@_profile_filename
@_fail_when_infeasible
def run(
    model_file,
    scenario,
    model_format,
    override_dict,
    save_netcdf,
    save_csv,
    save_logs,
    save_lp,
    debug,
    quiet,
    pdb,
    profile,
    profile_filename,
    fail_when_infeasible,
):
    """Execute the given model.

    Tries to guess from the file extension whether
    ``model_file`` is a YAML file or a pre-built model saved to NetCDF.
    This can also explicitly be set with the --model_format=yaml or
    --model_format=netcdf option.
    """
    start_time = _cli_start(debug, quiet)
    click.secho(
        f"Calliope {__version__} starting at {start_time.strftime(_time_format)}\n"
    )

    with format_exceptions(debug, pdb, profile, profile_filename, start_time):
        model = _run_setup_model(model_file, scenario, model_format, override_dict)
        click.secho(model.info() + "\n")

        # Only save LP file
        if save_lp:  # Only save LP file without solving model
            click.secho("Saving model to LP file...")
            if save_csv is not None or save_netcdf is not None:
                click.secho(
                    "WARNING: Model will not be solved - ignoring other save options!",
                    fg="red",
                    bold=True,
                )
            model.build()
            model.backend.to_lp(save_lp)
            print_end_time(start_time)

        # Else run the model, then save outputs
        else:
            click.secho("Starting model run...")
            kwargs = {}
            if save_logs:
                kwargs["solve.save_logs"] = save_logs

            if save_csv is None and save_netcdf is None:
                click.secho(
                    "\n!!!\nWARNING: No options to save results have been "
                    "specified.\nModel will run without saving results!\n!!!\n",
                    fg="red",
                    bold=True,
                )

            if (
                save_netcdf
                and model.config.solve.spores.save_per_spore_path is not None
            ):
                # If save_netcdf is used, override the 'save_per_spore_path' to point to
                # a directory of the same name as the planned netcdf
                kwargs["solve.spores.save_per_spore_path"] = (Path(save_netcdf).parent,)

            model.build()
            model.solve(**kwargs)
            termination = model.runtime.termination_condition

            if save_csv:
                click.secho(f"Saving CSV results to directory: {save_csv}")
                model.to_csv(save_csv)
            if save_netcdf:
                click.secho(f"Saving NetCDF results to file: {save_netcdf}")
                model.to_netcdf(save_netcdf)

            print_end_time(start_time)
            if fail_when_infeasible and termination != "optimal":
                raise BackendError("Problem is infeasible.")


@cli.command(
    name="generate_runs", short_help="Generate a script to run multiple models."
)
@click.argument("model_file")
@click.argument("out_file")
@click.option("--kind", help='One of: "bash", "bsub", "sbatch", or "windows".')
@click.option("--scenarios")
@click.option("--cluster_threads", default=1)
@click.option("--cluster_mem")
@click.option("--cluster_time")
@click.option(
    "--additional_args",
    default="",
    help="Any additional arguments to pass directly on to `calliope run`.",
)
@click.option("--override_dict")
@_debug
@_quiet
@_pdb
def generate_runs(
    model_file,
    out_file,
    kind,
    scenarios,
    additional_args,
    override_dict,
    cluster_threads,
    cluster_mem,
    cluster_time,
    debug,
    quiet,
    pdb,
):
    """Generates a script to run multiple models."""
    _cli_start(debug, quiet)

    kwargs = dict(
        model_file=model_file,
        out_file=out_file,
        scenarios=scenarios,
        additional_args=additional_args,
        override_dict=override_dict,
        cluster_mem=cluster_mem,
        cluster_time=cluster_time,
        cluster_threads=cluster_threads,
    )

    with format_exceptions(debug, pdb):
        generate(kind, **kwargs)


@cli.command(
    name="generate_scenarios",
    short_help="Generate scenario definitions from given combinations of overrides.",
)
@click.argument("model_file")
@click.argument("out_file")
@click.argument("overrides", nargs=-1)
@click.option("--scenario_name_prefix")
@_debug
@_quiet
@_pdb
def generate_scenarios(
    model_file, out_file, overrides, scenario_name_prefix, debug, quiet, pdb
):
    """Generate scenario definitions from given combinations of overrides."""
    _cli_start(debug, quiet)

    with format_exceptions(debug, pdb):
        combinations = list(itertools.product(*[i.split(";") for i in overrides]))

        if not scenario_name_prefix:
            scenario_name_prefix = "scenario_"

        # len(str(x)) gives us the number of digits in x, for padding
        scenario_string = "{}{:0>" + str(len(str(len(combinations)))) + "d}"

        scenarios = {
            "scenarios": {
                scenario_string.format(scenario_name_prefix, i + 1): c
                for i, c in enumerate(combinations)
            }
        }

        io.to_yaml(scenarios, path=out_file)
