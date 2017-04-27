"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
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

from . import core
from . import examples
from . import _version
from .parallel import Parallelizer


_debug = click.option(
    '--debug', is_flag=True, default=False,
    help='Print debug information when encountering errors.'
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

logging.basicConfig(stream=sys.stderr,
                    format='[%(asctime)s] %(levelname)-8s %(message)s',
                    datefmt=core._time_format)
logger = logging.getLogger()


@contextlib.contextmanager
def format_exceptions(debug=False, pdb=False, profile=False, profile_filename=None, start_time=None):
    if debug:
        logger.setLevel('DEBUG')

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


def print_end_time(start_time, msg='complete'):
    end_time = datetime.datetime.now()
    secs = round((end_time - start_time).total_seconds(), 1)
    tend = end_time.strftime(core._time_format)
    print('\nCalliope run {}. '
          'Elapsed: {} seconds (time at exit: {})'.format(msg, secs, tend))


def _get_version():
    return 'Version {}'.format(_version.__version__)


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


@cli.command(short_help='create model')
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
    if template is None:
        template = 'NationalScale'
    source_path = examples.PATHS[template]
    click.echo('Copying {} template to target directory: {}'.format(template, path))
    shutil.copytree(source_path, path)


@cli.command(short_help='directly run single model')
@click.argument('run_config')
@_debug
@_pdb
@_profile
@_profile_filename
def run(run_config, debug, pdb, profile, profile_filename):
    """Execute the given RUN_CONFIG run configuration file."""
    if debug:
        print(_get_version())
    logging.captureWarnings(True)
    start_time = datetime.datetime.now()
    with format_exceptions(debug, pdb, profile, profile_filename, start_time):
        tstart = start_time.strftime(core._time_format)
        print('Calliope run starting at {}\n'.format(tstart))
        model = core.Model(config_run=run_config)
        model.verbose = True  # Enables some print calls inside Model
        model_name = model.config_model.get_key('name', default='None')
        run_name = model.config_run.get_key('name', default='None')
        print('Model name:   {}'.format(model_name))
        print('Run name:     {}'.format(run_name))
        num_techs = (len(model.config_model.techs)
                     - len(core.get_default_techs()))
        msize = '{x} locations, {y} technologies, {t} timesteps'.format(
            x=len(model._sets['x']),
            y=num_techs,
            t=len(model._sets['t']))
        print('Model size:   {}\n'.format(msize))
        if not profile:
            model.config_run.set_key('output.save', True)  # Always save output
        model.run()
        print_end_time(start_time)


@cli.command(short_help='generate parallel runs')
@click.argument('run_config')
@click.argument('path', default='runs')
@click.option('--silent', is_flag=True, default=False,
              help='Be less verbose.')
@_debug
@_pdb
def generate(run_config, path, silent, debug, pdb):
    """
    Generate parallel runs based on the given RUN_CONFIG configuration
    file, saving them in the given PATH, which is a path to a
    directory that must not yet exist (PATH defaults to 'runs'
    if not specified).
    """
    if debug:
        print(_get_version())
    logging.captureWarnings(True)
    with format_exceptions(debug, pdb):
        parallelizer = Parallelizer(target_dir=path, config_run=run_config)
        if not silent and 'name' not in parallelizer.config.parallel:
            click.echo('`' + run_config +
                       '` does not specify a `parallel.name`' +
                       'and was skipped.')
            return
        click.echo('Generating runs from config '
                   '`{}` inside `{}`'.format(run_config, path))
        parallelizer.generate_runs()
