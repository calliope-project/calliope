"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

cli.py
~~~~~~

Command-line interface.

"""

import contextlib
import logging
import os
import shutil
import sys
import traceback

import click

from . import core
from .parallel import Parallelizer


_debug = click.option('--debug', is_flag=True, default=False,
                      help='Print debug information when encountering errors.')


@contextlib.contextmanager
def format_exceptions(debug=False):
    try:
        yield
    except Exception as e:
        if debug:
            traceback.print_exc()
        else:
            stack = traceback.extract_tb(e.__traceback__)
            # Get last stack trace entry still in Calliope
            last = [i for i in stack if 'calliope' in i[0]][-1]
            err_string = 'Error in {}, {}:{}'.format(last[2], last[0], last[1])
            click.secho(err_string, fg='red')
            click.secho('\n' + str(e) + '\n')
        sys.exit(1)


@click.group()
def cli():
    """Calliope: a multi-scale energy systems (MUSES) modeling framework"""
    pass


@cli.command(short_help='create model')
@click.argument('path')
def new(path):
    """
    Create new model at the given PATH, based on the included example model.
    The directory must not yet exist, and intermediate directories will
    be created automatically.
    """
    # Copies the included example model
    example_model = os.path.join(os.path.dirname(__file__), 'example_model')
    click.echo('Creating new model at: {}'.format(path))
    shutil.copytree(example_model, path)


@cli.command(short_help='directly run single model')
@click.argument('run_config')
@_debug
def run(run_config, debug):
    """Execute the given RUN_CONFIG run configuration file."""
    logging.captureWarnings(True)
    with format_exceptions(debug):
        model = core.Model(config_run=run_config)
        model.config_run.set_key('output.save', True)  # Always save output
        model.run()


@cli.command(short_help='generate parallel runs')
@click.argument('run_config')
@click.argument('path', default='runs')
@click.option('--silent', is_flag=True, default=False,
              help='Be less verbose.')
@_debug
def generate(run_config, path, silent, debug):
    """
    Generate parallel runs based on the given RUN_CONFIG configuration
    file, saving them in the given PATH, which is a path to a
    directory that must not yet exist (PATH defaults to 'runs'
    if not specified).
    """
    logging.captureWarnings(True)
    with format_exceptions(debug):
        parallelizer = Parallelizer(target_dir=path, config_run=run_config)
        if not silent and 'name' not in parallelizer.config.parallel:
            click.echo('`' + run_config +
                       '` does not specify a `parallel.name`' +
                       'and was skipped.')
            return
        click.echo('Generating runs from config '
                   '`{}` at `{}`'.format(run_config, path))
        parallelizer.generate_runs()
