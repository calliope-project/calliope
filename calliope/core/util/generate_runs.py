"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_runs.py
~~~~~~~~~~~~~~~~

Generate scripts to run multiple versions of the same model
in parallel on a cluster or sequentially on any machine.

"""

import os

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

    with open(out_file, 'w') as f:
        f.write('\n'.join(lines_all))

    os.chmod(out_file, 0o755)

    return commands


def generate_bsub_script(out_file, model_file, override_file, groups, additional_args, cluster_mem, cluster_time, cluster_threads=1, **kwargs):

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

    with open(out_file, 'w') as f:
        f.write('\n'.join(lines))


def generate_windows_script(out_file, model_file, override_file, groups, additional_args=None, **kwargs):
    commands = generate_runs(model_file, override_file, groups, additional_args)

    # \r\n are Windows line endings
    base_string = 'echo "Run {i}"\r\n{cmd}\r\n'
    lines_start = [
        '@echo off',
        '',
    ]

    lines_all = lines_start + [base_string.format(i=i + 1, cmd=cmd) for i, cmd in enumerate(commands)]

    with open(out_file, 'w') as f:
        f.write('\r\n'.join(lines_all))

    os.chmod(out_file, 0o755)

    return commands


_KINDS = {
    'bash': generate_bash_script,
    'bsub': generate_bsub_script,
    'windows': generate_windows_script,
}


def generate(kind, **kwargs):
    _KINDS[kind](**kwargs)
