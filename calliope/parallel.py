from __future__ import print_function
from __future__ import division

import argparse
import itertools
import os
import shutil
import sys

import pandas as pd
import yaml

from . import utils


class Parallelizer(object):
    """
    Args:
        additional_lines : can override additional_lines setting from
                           the parallel_settings.yaml file.

    """
    def __init__(self, config=None, target_dir=None, additional_lines=None):
        super(Parallelizer, self).__init__()
        self.config_file = config
        self.config = utils.AttrDict(yaml.load(open(config, 'r')))
        self.target_dir = target_dir
        if additional_lines:
            self.config.additional_lines = additional_lines

    def generate_iterations(self):
        keys = (['override.' + c for c in self.config.iterations_model.keys()]
                + self.config.iterations_run.keys())
        values = (self.config.iterations_model.values()
                  + self.config.iterations_run.values())
        df = pd.DataFrame(list(itertools.product(*values)), columns=keys)
        return df

    def _write_modelcommands(self, f, settings):
        pth = os.path.join('Runs', settings)
        f.write('python -c "import calliope\n')
        f.write('model = calliope.Model(config_run=\'{}\')\n'.format(pth))
        f.write('model.run()"\n')

    def generate_runs(self):
        c = self.config
        # Create output directory
        if self.target_dir:
            out_dir = os.path.join(self.target_dir, c.name)
        else:
            out_dir = c.name
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, 'Runs'))
        os.makedirs(os.path.join(out_dir, 'Logs'))
        iterations = self.generate_iterations()
        # Save run settings in out_dir to figure out what all the numbers mean!
        os.makedirs(os.path.join(out_dir, 'Output'))
        iterations.to_csv(os.path.join(out_dir, 'Output', 'iterations.csv'))
        shutil.copy(self.config_file, os.path.join(out_dir, 'Output',
                                                   'parallel_settings.yaml'))
        #
        # SET UP ADDITIONAL FILES FOR ARRAY RUNS
        #
        if c.style == 'array':
            array_submission = 'run.sh'
            array_run = 'array_run.sh'
            # Write array submission script
            with open(os.path.join(out_dir, array_submission), 'w') as f:
                f.write('#!/bin/sh\n')
                if c.environment == 'qsub':
                    f.write('#$ -t 1-{}\n'.format(len(iterations)))
                    f.write('#$ -N {}\n'.format(c.name))
                    f.write('#$ -j y -o Logs/array_$JOB_ID.log\n')
                    f.write('#$ -cwd\n')
                    f.write('\n./{} '.format(array_run) + '${SGE_TASK_ID}\n\n')
                elif c.environment == 'bsub':
                    f.write('#BSUB -J {}[1-{}]\n'.format(c.name,
                                                         len(iterations)))
                    t = 'run_settings.resources.memory'
                    if c.get_key(t, default=False):
                        f.write('#BSUB -R "rusage[mem={}]"\n'.format(c.get_key(t)))
                    t = 'run_settings.resources.threads'
                    if c.get_key(t, default=False):
                        f.write('#BSUB -n {}\n'.format(c.get_key(t)))
                    t = 'run_settings.resources.wall_time'
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
        for row in iterations.iterrows():
            # Generate configuration object
            index, item = row
            index_str = '{:0>4d}'.format(index)
            iteration_config = c.run_settings
            for k, v in item.to_dict().iteritems():
                iteration_config.set_key(k, v)
            # Always make sure we are saving outputs from iterations!
            iteration_config.set_key('output.save', True)
            # Set output dir in configuration object (NB: relative path!)
            iteration_config.set_key('output.path', os.path.join('Output',
                                                                 index_str))
            # Write configuration object to YAML file
            settings = 'settings_{}.yaml'.format(index_str)
            with open(os.path.join(out_dir, 'Runs', settings), 'w') as f:
                yaml.dump(iteration_config.as_dict(), f)
            if c.style == 'single':
                # Write model run script
                run = 'run_{}.sh'.format(index_str)
                with open(os.path.join(out_dir, run), 'w') as f:
                    f.write('#!/bin/sh\n')
                    if c.additional_lines:
                        f.write(c.additional_lines + '\n')
                    # Set job name
                    if c.environment == 'bsub':
                        if c.get_key('run_settings.limit_memory',
                                     default=False):
                            mem = c.get_key('run_settings.limit_memory')
                            f.write('#BSUB -R "rusage[mem={}]"\n'.format(mem))
                        f.write('#BSUB -J {}{}\n'.format(c.name, index_str))
                    # Write model commands
                    self._write_modelcommands(f, settings)
                os.chmod(os.path.join(out_dir, run), 0755)
            elif c.style == 'array':
                with open(os.path.join(out_dir, array_run), 'a') as f:
                    # Write index + 1 because the array jobs are 1-indexed
                    f.write('{}) '.format(index + 1))
                    if c.additional_lines:
                        f.write(c.additional_lines + '\n')
                    self._write_modelcommands(f, settings)
                    f.write(';;\n\n')
        # Final tasks after going through all iterations
        if c.style == 'array':
            with open(os.path.join(out_dir, array_run), 'a') as f:
                f.write('esac\n')
        os.chmod(os.path.join(out_dir, array_submission), 0755)
        os.chmod(os.path.join(out_dir, array_run), 0755)


def main():
    arguments = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Run the Calliope model.')
    parser.add_argument('settings', metavar='settings', type=str, default='',
                        help='parallel_settings file to use')
    parser.add_argument('-s', '--single', dest='single', action='store_const',
                        const=True, default=False,
                        help='don\'t do a parallel run, '
                             'interpret settings file as '
                             'run_settings instead of parallel_settings')
    parser.add_argument('-d', '--dir', type=str, default='runs',
                        help='target directory (default: runs)')
    args = parser.parse_args(arguments)
    parallelizer = Parallelizer(args.settings, target_dir='runs')
    parallelizer.generate_runs()


if __name__ == '__main__':
    main()
