from __future__ import print_function
from __future__ import division

import itertools
import os
import shutil

import pandas as pd
import yaml

from . import utils


class LisaParallelizer(object):
    """
    Args:
        additional_lines : can override additional_lines setting from
                           the parallel_settings.yaml file.

    """
    def __init__(self, config=None, target_dir=None, additional_lines=None):
        super(LisaParallelizer, self).__init__()
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
        f.write('python -c "import lisa\n')
        f.write('model = lisa.Lisa(config_run=\'{}\')\n'.format(settings))
        f.write('model.run()"\n')

    def generate_runs(self):
        c = self.config
        # Create output directory
        if self.target_dir:
            out_dir = os.path.join(self.target_dir, c.name)
        else:
            out_dir = c.name
        os.makedirs(out_dir)
        iterations = self.generate_iterations()
        # Save run settings in out_dir to figure out what all the numbers mean!
        os.makedirs(os.path.join(out_dir, 'Output'))
        iterations.to_csv(os.path.join(out_dir, 'Output', 'iterations.csv'))
        shutil.copy(self.config_file, os.path.join(out_dir, 'Output',
                                                   'parallel_settings.yaml'))
        if c.style == 'array':
            array_submitter = 'array_submitter.sh'
            array_submission = 'array_submission.sh'
            array_run = 'array_run.sh'
            # Write array submitter script
            with open(os.path.join(out_dir, array_submitter), 'w') as f:
                if c.environment == 'qsub':
                    f.write('#! /bin/sh\n')
                    f.write('qsub -t 1-{} -N {} {}\n'.format(len(iterations),
                                                             c.name,
                                                             array_submission))
            # Write array submission script
            with open(os.path.join(out_dir, array_submission), 'w') as f:
                f.write('#!/bin/sh\n')
                if c.environment == 'qsub':
                    f.write('#$ -j y -o Output/array_$JOB_ID.log\n')
                    f.write('#$ -cwd\n')
                    f.write('./{} '.format(array_run) +
                            '${JOB_ID} ${SGE_TASK_ID}\n')
            # Set up array run script
            with open(os.path.join(out_dir, array_run), 'w') as f:
                f.write('#!/bin/sh\n')
                f.write('case "$2" in\n\n')
        # Write settings and scripts for each iteration
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
            with open(os.path.join(out_dir, settings), 'w') as f:
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
