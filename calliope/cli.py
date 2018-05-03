"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

cli.py
~~~~~~

Command-line interface.

"""

import contextlib
import datetime
import logging
import os
import pstats
import shutil
import sys
import traceback
import click

from calliope import Model, read_netcdf, examples
from calliope.core.util.convert import convert_model
from calliope.core.util.generate_runs import generate
from calliope.core.util.logging import logger, set_log_level
from calliope._version import __version__


_time_format = '%Y-%m-%d %H:%M:%S'


_debug = click.option(
    '--debug', is_flag=True, default=False,
    help='Print debug information when encountering errors.'
)

_quiet = click.option(
    '--quiet', is_flag=True, default=False,
    help='Be less verbose about what is happening, including hiding '
         'solver output.'
)

_pdb = click.option(
    '--pdb', is_flag=True, default=False,
    help='If used together with --debug, drop into interactive '
         'debugger on encountering errors.'
)

_profile = click.option(
    '--profile', is_flag=True, default=False,
    help='Run through cProfile.'
)

_profile_filename = click.option(
    '--profile_filename', type=str,
    help='Filename to save profile to if enabled --profile.'
)

if logger.hasHandlers():
    for handler in logger.handlers:
        logger.removeHandler(handler)

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)-8s %(message)s', datefmt=_time_format
)
console = logging.StreamHandler(stream=sys.stderr)
console.setFormatter(formatter)
logger.addHandler(console)

@contextlib.contextmanager
def format_exceptions(debug=False, pdb=False, profile=False, profile_filename=None, start_time=None):
    if debug:
        set_log_level('DEBUG')

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
                print('\nSaving cProfile output to: {}'.format(dump_path))
                profile.dump_stats(dump_path)
            else:
                print('\n\n----PROFILE OUTPUT----\n\n')
                stats = pstats.Stats(profile).sort_stats('cumulative')
                stats.print_stats(20)  # Print first 20 lines

    except Exception as e:
        if debug:
            traceback.print_exc()
            if pdb:
                import pdb
                pdb.post_mortem(e.__traceback__)
        else:
            stack = traceback.extract_tb(e.__traceback__)
            # Get last stack trace entry still in Calliope
            last = [i for i in stack if 'calliope' in i[0]][-1]
            if debug:
                err_string = '\nError in {}, {}:{}'.format(last[2], last[0], last[1])
            else:
                err_string = '\nError in {}:'.format(last[2])
            click.secho(err_string, fg='red')
            click.secho(str(e), fg='red')
            if start_time:
                print_end_time(start_time, msg='aborted due to an error')
        sys.exit(1)


def set_quietness_level(quiet):
    if quiet:
        set_log_level('WARNING')
    else:
        set_log_level('SOLVER')


def print_end_time(start_time, msg='complete'):
    end_time = datetime.datetime.now()
    secs = round((end_time - start_time).total_seconds(), 1)
    tend = end_time.strftime(_time_format)
    print('\nCalliope run {}. '
          'Elapsed: {} seconds (time at exit: {})'.format(msg, secs, tend))


def _get_version():
    return 'Version {}'.format(__version__)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', is_flag=True, default=False,
              help='Display version.')
def cli(ctx, version):
    """Calliope: a multi-scale energy systems (MUSES) modeling framework"""
    if ctx.invoked_subcommand is None and not version:
        print(ctx.get_help())
    if version:
        print(_get_version())


@cli.command(short_help='Create a new model based on a built-in example.')
@click.argument('path')
@click.option('--template', type=str, default=None)
@_debug
def new(path, template, debug):
    """
    Create new model at the given ``path``, based on one of the built-in
    example models. The target path must not yet exist. Intermediate
    directories will be created automatically.
    """
    if debug:
        print(_get_version())
    with format_exceptions(debug):
        if template is None:
            template = 'national_scale'
        source_path = examples._PATHS[template]
        click.echo('Copying {} template to target directory: {}'.format(template, path))
        shutil.copytree(source_path, path)


@cli.command(short_help='Run a model.')
@click.argument('model_file')
@click.option('--override_file')
@click.option('--save_netcdf')
@click.option('--save_csv')
@click.option('--save_plots')
@click.option('--save_logs')
@click.option('--model_format')
@_debug
@_quiet
@_pdb
@_profile
@_profile_filename
def run(model_file, override_file, save_netcdf, save_csv, save_plots,
        save_logs, model_format,
        debug, quiet, pdb, profile, profile_filename):
    """
    Execute the given model. Tries to guess from the file extension whether
    ``model_file`` is a YAML file or a pre-built model saved to NetCDF.
    This can also explicitly be set with the --model_format=yaml or
    --model_format=netcdf option.

    """
    if debug:
        print(_get_version())

    set_quietness_level(quiet)

    logging.captureWarnings(True)
    pywarning_logger = logging.getLogger('py.warnings')
    pywarning_logger.addHandler(console)

    start_time = datetime.datetime.now()
    with format_exceptions(debug, pdb, profile, profile_filename, start_time):
        if save_csv is None and save_netcdf is None:
            click.secho(
                '\n!!!\nWARNING: No options to save results have been '
                'specified.\nModel will run without saving results!\n!!!\n',
                fg='red', bold=True
            )
        tstart = start_time.strftime(_time_format)
        print('Calliope run starting at {}\n'.format(tstart))

        # Try to determine model file type if not given explicitly
        if model_format is None:
            if model_file.split('.')[-1] in ['yaml', 'yml']:
                model_format = 'yaml'
            elif model_file.split('.')[-1] in ['nc', 'nc4', 'netcdf']:
                model_format = 'netcdf'
            else:
                raise ValueError(
                    'Cannot determine model file format based on file '
                    'extension for "{}". Set format explicitly with '
                    '--model_format.'.format(model_file)
                )

        if model_format == 'yaml':
            override_dict = {
                'run.save_logs': save_logs
            }
            model = Model(
                model_file, override_file=override_file, override_dict=override_dict
            )
        elif model_format == 'netcdf':
            if override_file is not None:
                raise ValueError(
                    'Overrides cannot be applied when loading a pre-built '
                    'model from NetCDF. Please run without --override options.'
                )
            model = read_netcdf(model_file)
            if save_logs is not None:
                model._model_data.attrs['run.save_logs'] = save_logs
        else:
            raise ValueError('Invalid model format: {}'.format(model_format))

        print(model.info() + '\n')
        print('Starting model run...')
        model.run()

        termination = model._model_data.attrs.get(
            'termination_condition', 'unknown')
        if save_csv:
            print('Saving CSV results to directory: {}'.format(save_csv))
            model.to_csv(save_csv)
        if save_netcdf:
            print('Saving NetCDF results to file: {}'.format(save_netcdf))
            model.to_netcdf(save_netcdf)
        if save_plots:
            if termination == 'optimal':
                print('Saving HTML file with plots to: {}'.format(save_plots))
                model.plot.summary(to_file=save_plots)
            else:
                click.secho(
                    'Model termination condition non-optimal. Not saving plots',
                    fg='red', bold=True
                )
        print_end_time(start_time)


@cli.command(short_help='Generate a script to run multiple models.')
@click.argument('model_file')
@click.argument('out_file')
@click.option('--kind', help='One of: "bash", "bsub".')
@click.option('--override_file')
@click.option('--groups')
@click.option('--cluster_threads', default=1)
@click.option('--cluster_mem')
@click.option('--cluster_time')
@click.option(
    '--additional_args', default='',
    help='Any additional arguments to pass directly on to `calliope run`.')
@_debug
@_quiet
@_pdb
def generate_runs(
        model_file, out_file, kind, override_file, groups, additional_args,
        cluster_threads, cluster_mem, cluster_time,
        debug, quiet, pdb):

    set_quietness_level(quiet)

    kwargs = dict(
        model_file=model_file,
        out_file=out_file,
        override_file=override_file,
        groups=groups,
        additional_args=additional_args,
        cluster_mem=cluster_mem,
        cluster_time=cluster_time,
        cluster_threads=cluster_threads,
    )

    with format_exceptions(debug, pdb):
        generate(kind, **kwargs)


@cli.command(short_help='Convert a 0.5.x model to 0.6.0.')
@click.argument('run_config_path')
@click.argument('model_config_path')
@click.argument('out_path')
@_debug
@_quiet
@_pdb
def convert(
        run_config_path, model_config_path, out_path,
        debug, quiet, pdb):
    """
    Convert a Calliope 0.5.x model specified by its run and model
    configurations to a Calliope 0.6.0 model.

    The input model is not modified; the converted version of the model
    is saved to ``out_path``.

    """
    set_quietness_level(quiet)
    with format_exceptions(debug, pdb):
        print('Converting model...')
        convert_model(run_config_path, model_config_path, out_path)
        print('Done. Results saved to: {}'.format(out_path))
