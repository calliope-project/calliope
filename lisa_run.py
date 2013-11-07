#!/usr/bin/env python

import argparse
import sys

import lisa


arguments = sys.argv[1:]
parser = argparse.ArgumentParser(description='Run the Lisa model.')
parser.add_argument('settings', metavar='settings', type=str, default='',
                    help='parallel_settings file to use')
parser.add_argument('-s', '--single', dest='single', action='store_const',
                    const=True, default=False,
                    help='don\'t do a parallel run, interpret settings file as'
                    ' run_settings instead of parallel_settings')
parser.add_argument('-d', '--dir', type=str, default='runs',
                    help='target directory (default: runs)')
args = parser.parse_args(arguments)

parallelizer = lisa.Parallelizer(args.settings, target_dir='runs')
parallelizer.generate_runs()
