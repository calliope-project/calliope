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
    overrides = AttrDict.from_yaml(override_file)

    if groups is None:
        runs = overrides.keys()
    else:
        runs = groups.split(';')

    commands = []

    for i, run in enumerate(runs):
        cmd = 'calliope run {model} --override_file {override}:{groups} --save_netcdf out_{i}_{groups}.nc {other_options}'.format(
            i=i+1,
            model=model_file,
            override=override_file,
            groups=run,
            other_options=additional_args
        ).strip()
        commands.append(cmd)

    return commands


def generate_bash_script(out_file, model_file, override_file, groups, additional_args=None):
    base_string = '{i}) {cmd} ;;\n'
    lines_start = ['#!/bin/sh', '', 'case "$1" in', '']
    lines_end = ['esac', '']

    commands = generate_runs(model_file, override_file, groups, additional_args)

    lines_all = lines_start + [base_string.format(i=i+1, cmd=cmd) for i, cmd in enumerate(commands)] + lines_end

    with open(out_file, 'w') as f:
        f.write('\n'.join(lines_all))

    os.chmod(out_file, 0o755)

    return commands


def generate_bsub_script(out_file, model_file, override_file, groups, additional_args, cluster_mem, cluster_time, cluster_threads=1):
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


_KINDS = {
    'bash': generate_bash_script,
    'bsub': generate_bsub_script
    # 'windows': generate_windows_script,
}


def generate(kind, **kwargs):
    _KINDS[kind](**kwargs)
