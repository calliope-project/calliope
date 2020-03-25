import argparse
import subprocess

import ruamel_yaml as yaml


def conda_create_command(
    requirement_files,
    env_name="calliope",
    python_version="3",
    channels=[],
    ignore=[],
    run=False,
):
    """
    Parameters
    ----------

    requirements_files : list
    env_name : str, optional
    python_version: str, optional
    channels: list, optional
    ignore : list, optional
    run : bool, optional

    Returns
    -------
    cmd : str

    """

    chan_set = set()
    dep_set = set()

    for req in requirement_files:
        with open(req) as f:
            y = yaml.safe_load(f)

        # Filter the optional `pip` dict in dependencies
        str_deps = list(filter(lambda i: isinstance(i, str), y["dependencies"]))

        if ignore:
            # Filter ignored dependencies
            str_deps = list(
                filter(lambda i: not any([ign in i for ign in ignore]), str_deps)
            )

        dep_set.update(str_deps)
        chan_set.update(y["channels"])

    dep_string = " ".join(['"{}"'.format(i) for i in sorted(dep_set)])

    if channels:
        chan_string = " ".join(["-c " + i for i in channels])
    else:
        chan_string = " ".join(["-c " + i for i in chan_set])

    cmd = 'conda create --name {name} --override-channels {chans} "python={py}" {deps}'.format(
        name=env_name, chans=chan_string, deps=dep_string, py=python_version
    )

    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("requirement_file", type=str, nargs="+")
    parser.add_argument("--env_name", type=str, default="calliope")
    parser.add_argument("--python_version", type=str, default="3")
    parser.add_argument("--channels", type=str, nargs="*")
    parser.add_argument("--ignore", type=str, nargs="*")
    parser.add_argument("--run", dest="run", action="store_true", default=False)

    args = parser.parse_args()

    cmd = conda_create_command(
        args.requirement_file,
        args.env_name,
        args.python_version,
        args.channels,
        args.ignore,
    )

    if args.run:
        print("Running: {}".format(cmd))
        subprocess.run(cmd, shell=True)
    else:
        print(cmd)
