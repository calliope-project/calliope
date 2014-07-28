"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

parallel.py
~~~~~~~~~~~

Defines the Parallelizer class which manages the creation of arbitrary
numbers of scripted runs, either locally or to be deployed to a cluster.

"""

from __future__ import print_function
from __future__ import division

import argparse
import copy
import itertools
import os
import sys

import numpy as np
import pandas as pd

from . import core
from . import exceptions
from . import utils


class Parallelizer(object):
    """Arguments:

    * ``target_dir``: path to output directory for parallel runs.
    * ``config_run``: path to YAML file with run settings. If not
      given, ``{{ module }}/config/run.yaml`` is used as the default.

    """
    def __init__(self, target_dir, config_run=None):
        super(Parallelizer, self).__init__()
        config_path = os.path.join(os.path.dirname(__file__), 'config')
        if not config_run:
            config_run = os.path.join(config_path, 'run.yaml')
        self.config_file = config_run
        self.config = utils.AttrDict.from_yaml(config_run)
        self.target_dir = target_dir

    def generate_iterations(self):
        c = self.config.parallel.iterations
        if isinstance(c, list):
            df = pd.DataFrame(c)
        elif isinstance(c, dict):
            iter_keys = c.keys_nested()
            iter_values = [c.get_key(i) for i in iter_keys]
            df = pd.DataFrame(list(itertools.product(*iter_values)),
                              columns=iter_keys)
        return df

    def _write_modelcommands(self, f, settings):
        pth = os.path.join('Runs', settings)
        f.write('python -c "import calliope\n')
        f.write('model = calliope.Model(config_run=\'{}\')\n'.format(pth))
        f.write('model.run()"\n')

    def _write_additional_lines(self, f):
        c = self.config
        lines = c.parallel.additional_lines
        if isinstance(lines, list):
            f.writelines([i + '\n' for i in lines])
        else:
            f.write(lines + '\n')

    def generate_runs(self):
        c = self.config
        # Create output directory
        if self.target_dir:
            out_dir = os.path.join(self.target_dir, c.parallel.name)
        else:
            out_dir = c.parallel.name
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, 'Runs'))
        os.makedirs(os.path.join(out_dir, 'Logs'))
        iterations = self.generate_iterations()
        # Save relevant settings in out_dir,
        # to figure out what all the numbers mean!
        os.makedirs(os.path.join(out_dir, 'Output'))
        iterations.to_csv(os.path.join(out_dir, 'Output', 'iterations.csv'))
        parallel_f = os.path.join(out_dir, 'Output', 'parallel_settings.yaml')
        c.parallel.to_yaml(parallel_f)

        #
        # COMBINE ALL MODEL CONFIG INTO ONE FILE AND WRITE IT TO out_dir
        #
        o = core.get_model_config(c, self.config_file, adjust_data_path=False)
        unified_config_file = os.path.join(out_dir, 'Runs', 'model.yaml')
        o.to_yaml(os.path.join(unified_config_file))
        c.input.model = 'model.yaml'

        #
        # SET UP ADDITIONAL FILES FOR ARRAY RUNS
        #
        if c.parallel.style == 'array':
            array_submission = 'run.sh'
            array_run = 'array_run.sh'
            # Write array submission script
            with open(os.path.join(out_dir, array_submission), 'w') as f:
                f.write('#!/bin/sh\n')
                if c.parallel.environment == 'qsub':
                    f.write('#$ -t 1-{}\n'.format(len(iterations)))
                    f.write('#$ -N {}\n'.format(c.parallel.name))
                    f.write('#$ -j y -o Logs/array_$JOB_ID.log\n')
                    t = 'parallel.resources.memory'
                    if c.get_key(t, default=False):
                        mem_gb = c.get_key(t) / 1000.0
                        f.write('#$ -l mem_total={:.1f}G\n'.format(mem_gb))
                    t = 'parallel.resources.threads'
                    if c.get_key(t, default=False):
                        f.write('#$ -pe smp {}\n'.format(c.get_key(t)))
                    f.write('#$ -cwd\n')
                    f.write('\n./{} '.format(array_run) + '${SGE_TASK_ID}\n\n')
                elif c.parallel.environment == 'bsub':
                    f.write('#BSUB -J {}[1-{}]\n'.format(c.parallel.name,
                                                         len(iterations)))
                    t = 'parallel.resources.memory'
                    if c.get_key(t, default=False):
                        f.write('#BSUB -R "rusage[mem={}]"\n'.format(c.get_key(t)))
                    t = 'parallel.resources.threads'
                    if c.get_key(t, default=False):
                        f.write('#BSUB -n {}\n'.format(c.get_key(t)))
                    t = 'parallel.resources.wall_time'
                    if c.get_key(t, default=False):
                        f.write('#BSUB -W {}\n'.format(c.get_key(t)))
                    f.write('#BSUB -o Logs/lsf_%I.log\n')
                    f.write('\n./{} '.format(array_run) + '${LSB_JOBINDEX}\n\n')
            # Set up array run script
            with open(os.path.join(out_dir, array_run), 'w') as f:
                f.write('#!/bin/sh\n')
                f.write('case "$1" in\n\n')
        #
        # WRITE SETTINGS AND SCRIPTS FOR EACH ITERATION
        #
        # Always make sure we are saving outputs from iterations!
        c.set_key('output.save', True)
        for row in iterations.iterrows():
            iter_c = copy.copy(c)  # iter_c is this iteration's config
            # Generate configuration object
            index, item = row
            index_str = '{:0>4d}'.format(index)
            for k, v in item.to_dict().iteritems():
                # Convert numpy dtypes to python ones, else YAML chokes
                if isinstance(v, np.generic):
                    v = np.asscalar(v)
                iter_c.set_key(k, copy.copy(v))
            # Set output dir in configuration object, this is hardcoded
            iter_c.set_key('output.path', os.path.join('Output', index_str))
            # Remove parallel key, not needed in individual files
            del iter_c.parallel
            # Write configuration object to YAML file
            settings = 'settings_{}.yaml'.format(index_str)
            iter_c.to_yaml(os.path.join(out_dir, 'Runs', settings))
            if c.parallel.style == 'single':
                # Write model run script
                run = 'run_{}.sh'.format(index_str)
                with open(os.path.join(out_dir, run), 'w') as f:
                    f.write('#!/bin/sh\n')
                    if c.parallel.additional_lines:
                        self._write_additional_lines(f)
                    # Set job name
                    if c.parallel.environment == 'bsub':
                        f.write('#BSUB -J {}{}\n'.format(c.parallel.name,
                                                         index_str))
                    # Write model commands
                    self._write_modelcommands(f, settings)
                os.chmod(os.path.join(out_dir, run), 0755)
            elif c.parallel.style == 'array':
                with open(os.path.join(out_dir, array_run), 'a') as f:
                    # Write index + 1 because the array jobs are 1-indexed
                    f.write('{}) '.format(index + 1))
                    if c.parallel.additional_lines:
                        self._write_additional_lines(f)
                    self._write_modelcommands(f, settings)
                    f.write(';;\n\n')
        # Final tasks after going through all iterations
        if c.parallel.style == 'array':
            with open(os.path.join(out_dir, array_run), 'a') as f:
                f.write('esac\n')
            os.chmod(os.path.join(out_dir, array_submission), 0755)
            os.chmod(os.path.join(out_dir, array_run), 0755)


def main():
    arguments = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Run the Calliope model.')
    parser.add_argument('settings', metavar='settings', type=str, default='',
                        help='Run settings file to use.')
    parser.add_argument('-s', '--single', dest='single', action='store_const',
                        const=True, default=False,
                        help='Ignore `parallal` section in run settings, '
                             'and only do a single run.')
    parser.add_argument('-d', '--dir', type=str, default='runs',
                        help='Target directory for parallel runs '
                             '(default: `runs`).')
    args = parser.parse_args(arguments)
    if args.single:
        model = core.Model(config_run=args.settings)
        model.run()
    else:
        parallelizer = Parallelizer(target_dir=args.dir,
                                    config_run=args.settings)
        if not 'name' in parallelizer.config.parallel:
            raise exceptions.ModelError('`' + args.settings + '`'
                                        ' does not specify a `parallel.name`.')
        parallelizer.generate_runs()


if __name__ == '__main__':
    main()
