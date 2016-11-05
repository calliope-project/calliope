"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

cli.py
~~~~~~

Command-line interface.

"""

import contextlib
import datetime
import logging
import os
import shutil
import sys
import traceback

import click

from . import core
from . import _version
from .parallel import Parallelizer


_debug = click.option('--debug', is_flag=True, default=False,
                      help='Print debug information when encountering errors.')
_pdb = click.option('--pdb', is_flag=True, default=False,
                    help='If used together with --debug, drop into interactive '
                         'debugger on encountering errors.')


logging.basicConfig(stream=sys.stderr,
                    format='[%(asctime)s] %(levelname)-8s %(message)s',
                    datefmt=core._time_format)
logger = logging.getLogger()


@contextlib.contextmanager
def format_exceptions(debug=False, pdb=False, start_time=None):
    if debug:
        logger.setLevel('DEBUG')
    try:
        yield
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


def print_debug_startup(debug):
    if debug:
        print('Version {}'.format(_version.__version__))


@click.group()
def cli():
    """Calliope: a multi-scale energy systems (MUSES) modeling framework"""
    pass


@cli.command(short_help='create model')
@click.argument('path')
@_debug
def new(path, debug):
    """
    Create new model at the given PATH, based on the included example model.
    The directory must not yet exist, and intermediate directories will
    be created automatically.
    """
    print_debug_startup(debug)
    # Copies the included example model
    example_model = os.path.join(os.path.dirname(__file__), 'example_model')
    click.echo('Creating new model in: {}'.format(path))
    shutil.copytree(example_model, path)


@cli.command(short_help='directly run single model')
@click.argument('run_config')
@_debug
@_pdb
def run(run_config, debug, pdb):
    """Execute the given RUN_CONFIG run configuration file."""
    print_debug_startup(debug)
    logging.captureWarnings(True)
    start_time = datetime.datetime.now()
    with format_exceptions(debug, pdb, start_time):
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
    print_debug_startup(debug)
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
