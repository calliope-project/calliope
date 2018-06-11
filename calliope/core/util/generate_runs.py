"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_runs.py
~~~~~~~~~~~~~~~~

Generate scripts to run multiple versions of the same model
in parallel on a cluster or sequentially on any machine.

"""

import os

import pandas as pd
from calliope.core import AttrDict


def generate_runs(model_file, override_file, groups=None, additional_args=None):
    """
    Returns a list of "calliope run" invocations.

    groups specified as group1{,group2,...}{;group3...}

    if groups not specified, use all groups in the override_file, one by one

    """
    if groups is None:
        overrides = AttrDict.from_yaml(override_file)
        runs = overrides.keys()
    else:
        runs = groups.split(';')

    commands = []

    for i, run in enumerate(runs):
        cmd = 'calliope run {model} --override_file {override}:{groups} --save_netcdf out_{i}_{groups}.nc --save_plots plots_{i}_{groups}.html {other_options}'.format(
            i=i + 1,
            model=model_file,
            override=override_file,
            groups=run,
            other_options=additional_args
        ).strip()
        commands.append(cmd)

    return commands


def generate_bash_script(out_file, model_file, override_file, groups, additional_args=None, **kwargs):
    commands = generate_runs(model_file, override_file, groups, additional_args)

    base_string = '    {i}) {cmd} ;;\n'
    lines_start = [
        '#!/bin/sh',
        '',
        'function process_case () {',
        '    case "$1" in',
        ''
    ]
    lines_end = [
        '    esac', '}',
        '',
        'if [[ $# -eq 0 ]] ; then',
        '    echo No parameter given, running all runs sequentially...',
        '    for i in $(seq 1 {}); do process_case $i; done'.format(len(commands)),
        'else',
        '    echo Running run $1',
        '    process_case $1',
        'fi',
        '',
    ]

    lines_all = lines_start + [base_string.format(i=i + 1, cmd=cmd) for i, cmd in enumerate(commands)] + lines_end

    with open(out_file, 'wb') as f:
        f.write(bytes('\n'.join(lines_all), 'UTF-8'))

    os.chmod(out_file, 0o755)

    return commands


def generate_bsub_script(out_file, model_file, override_file, groups,
                         additional_args, cluster_mem, cluster_time,
                         cluster_threads=1, **kwargs):

    # We also need to generate the bash script to run on the cluster
    bash_out_file = out_file + '.array.sh'
    bash_out_file_basename = os.path.basename(bash_out_file)
    commands = generate_bash_script(bash_out_file, model_file, override_file, groups, additional_args)

    lines = [
        '#!/bin/sh',
        '#BSUB -J calliope[1-{}]'.format(len(commands)),
        '#BSUB -n {}'.format(cluster_threads),
        '#BSUB -R "rusage[mem={}]"'.format(cluster_mem),
        '#BSUB -W {}'.format(cluster_time),
        '#BSUB -r',  # Automatically restart failed jobs
        '#BSUB -o log_%I.log',
        '',
        './' + bash_out_file_basename + ' ${LSB_JOBINDEX}',
        ''
    ]

    with open(out_file, 'wb') as f:
        f.write(bytes('\n'.join(lines), 'UTF-8'))


def generate_sbatch_script(out_file, model_file, override_file, groups,
                           additional_args, cluster_mem, cluster_time,
                           cluster_threads=1, **kwargs):
    """
    SBATCH (SLURM) script generator.
    """

    # We also need to generate the bash script to run on the cluster
    bash_out_file = out_file + '.array.sh'
    bash_out_file_basename = os.path.basename(bash_out_file)
    commands = generate_bash_script(bash_out_file, model_file, override_file, groups, additional_args)

    if ':' not in cluster_time:
        # Assuming time given as minutes, so needs changing to %H:%M%S
        cluster_time = pd.to_datetime(cluster_time, unit='m').strftime('%H:%M:%S')

    lines = [
        '#!/bin/bash',
        '#SBATCH -J calliope',  # Name of the job
        '#SBATCH --array=1-{}'.format(len(commands)),  # How many jobs there are
        '#SBATCH --ntasks={}'.format(cluster_threads),
        '#SBATCH --mem={}'.format(cluster_mem),
        '#SBATCH --time={}'.format(cluster_time),  # How much wallclock time will be required
        '#SBATCH -o log_%a.log',
        '',
        '#! Optional add-ins for SBATCH (uncomment and add info as necessary):',
        '##SBATCH -A project_name',  # Which project should be charged'
        '##SBATCH --nodes=X',  # X whole nodes should be allocated'
        '##SBATCH -p partition_name',
        '',
        '#! Insert module load commands after this line, if needed:',
        '#! (Note: you can load this in ~.bashrc if you want them loaded every time you log in)',
        '',
        '#! module load gurobi (or glpk/cplex)',
        '#! module load miniconda3',
        '#! module load /path/to/miniconda3/envs/your_env_name/',
        '',
        'cd $SLURM_SUBMIT_DIR',
        '',
        './' + bash_out_file_basename + ' ${SLURM_ARRAY_TASK_ID}'
    ]

    with open(out_file, 'wb') as f:
        f.write(bytes('\n'.join(lines), 'UTF-8'))


def generate_windows_script(out_file, model_file, override_file, groups, additional_args=None, **kwargs):
    commands = generate_runs(model_file, override_file, groups, additional_args)

    # \r\n are Windows line endings
    base_string = 'echo "Run {i}"\r\n{cmd}\r\n'
    lines_start = [
        '@echo off',
        '',
    ]

    lines_all = lines_start + [base_string.format(i=i + 1, cmd=cmd) for i, cmd in enumerate(commands)]

    with open(out_file, 'wb') as f:
        f.write(bytes('\r\n'.join(lines_all), 'UTF-8'))

    os.chmod(out_file, 0o755)

    return commands


_KINDS = {
    'bash': generate_bash_script,
    'bsub': generate_bsub_script,
    'windows': generate_windows_script,
    'sbatch': generate_sbatch_script
}


def generate(kind, **kwargs):
    _KINDS[kind](**kwargs)
