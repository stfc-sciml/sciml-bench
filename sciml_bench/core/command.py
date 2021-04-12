#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# command.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Main entry of the program
"""

import click
from pathlib import Path
import sciml_bench.core.benchmark as Benchmark
import sciml_bench.core.dataset as Dataset
from sciml_bench.core.program import ProgramEnv
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.system import all_sys_info, format_info
from sciml_bench import __version__ as VERSION

# init a global ProgramEnv instance
SMLB_ENV = ProgramEnv(Path(__file__).parents[1] / 'sciml_bench_config.yml',
                      Path(__file__).parents[1] / 'benchmarks/registration.yml')


class NaturalOrderGroup(click.Group):
    """ Force click to keep the order of commands """

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=NaturalOrderGroup)
def cli():
    pass


@cli.command('list', help='List datasets and benchmarks.')
@click.option('--brief', is_flag=True, default=False,
              help='Use a brief layout. Default: False.')
@click.option('--verify', is_flag=True, default=False,
              help='\b\nVerify existence of datasets at default location'
                   '\nand verify modules of benchmarks. Default: False.')
@click.argument('scope', default='both',
                type=click.Choice(['both', 'datasets', 'benchmarks']))
def cmd_list(brief, verify, scope):
    """ sciml_bench list """
    # key width first
    width_data = len(max(list(SMLB_ENV.registered_datasets.keys()), key=len))
    width_ben = len(max(list(SMLB_ENV.registered_benchmarks.keys()), key=len))
    width = max(width_data, width_ben)

    # list datasets
    if scope == 'both' or scope == 'datasets':
        print('\nLIST OF DATASETS:')
        for name, prop in SMLB_ENV.registered_datasets.items():
            if brief:
                print(f'* {name.ljust(width + 4)} Size: {prop["size"]}')
            else:
                print(f'* {name}')
                print(f'  - Approximate size: {prop["size"]}')
                print(f'  - {prop["title"]}')
                print('  - {}'.format(prop["info"].replace('\n ', '\n').
                                      replace('\n', '\n    ')))  # indent
                if "s3.echo.stfc.ac.uk" in prop["download_method"]:
                    print('  - Remote data server: STFC')
                else:
                    print('  - Remote data server: by contributors')
                if verify:
                    found = Dataset.found(name, SMLB_ENV.dataset_root_dir)
                    found = 'Found' if found else 'Not Found'
                    print('  - {} in default dataset directory ({})'.
                          format(found, SMLB_ENV.dataset_root_dir))

    # list benchmarks
    if scope == 'both' or scope == 'benchmarks':
        print('\nLIST OF BENCHMARKS:')
        for name, prop in SMLB_ENV.registered_benchmarks.items():
            if brief:
                print(f'* {name.ljust(width + 4)} Dataset: {prop["dataset"]}')
            else:
                print(f'* {name}')
                print(f'  - Dataset: {prop["dataset"]}')
                print(f'  - {prop["title"]}')
                print('  - {}'.format(prop["info"].replace('\n ', '\n').
                                      replace('\n', '\n    ')))  # indent
                print(f'  - Dependencies: {prop["dependencies"]}')
                # find out if this benchmark is ready for run
                if verify:
                    good = Benchmark.create_instance(name, True) is not None
                    print(f'  - Modules verified: {good}')


@cli.command(help='Install benchmark dependencies.')
@click.argument('benchmark_list')
def install(benchmark_list):
    """ sciml_bench list """
    Benchmark.install_benchmark_dependencies(benchmark_list, SMLB_ENV)


@cli.command(help='Download a dataset.')
@click.option('--dataset_root_dir', default=SMLB_ENV.dataset_root_dir,
              help='\b\nRoot directory of datasets.\n'
                   'Default: dataset_root_dir in sciml_bench_config.yml.')
@click.argument('dataset_name')
def download(dataset_name, dataset_root_dir):
    """ sciml_bench download """
    dataset_dir = Dataset.download(dataset_name, Path(dataset_root_dir),
                                   SMLB_ENV)
    print(f'Dataset {dataset_name} downloaded to {dataset_dir}.')


@cli.command(help='Run a benchmark.')
@click.option('--dataset_dir', default='',
              help='\b\nDirectory of dataset.\n'
                   'Default: dataset_root_dir/dataset_name/\n'
                   '         (dataset_root_dir in sciml_bench_config.yml).')
@click.option('--output_dir', required=True,
              help='\b\nOutput directory of this run.\n'
                   'Convention: use --output_dir=@foo to save outputs under\n'
                   '            output_root_dir/benchmark_name/foo/;\n'
                   '            without "@", foo is used as a normal path.')
@click.option('--monitor_on/--monitor_off', default=True,
              help='\b\nMonitor system usage during runtime.'
                   '\nDefault: True.')
@click.option('--monitor_interval', default=1.0, type=float,
              help='\b\nTime interval for system monitoring.'
                   '\nDefault: 1.0.')
@click.option('--monitor_report_style', default='pretty',
              type=click.Choice(['pretty', 'yaml', 'hdf5']),
              help='\b\nReport style of system monitor.'
                   '\nDefault: pretty.')
@click.option('--benchmark_specific', '-b', 'bench_args_list',
              type=(str, str), multiple=True,
              help='\b\nBenchmark-specific arguments.\n'
                   'Usage: -b key1 val1 -b key2 val2 ...')
@click.argument('benchmark_name')
def run(dataset_dir, output_dir,
        monitor_on, monitor_interval, monitor_report_style,
        bench_args_list, benchmark_name):
    """ sciml_bench run """
    # runtime input
    smlb_in = RuntimeIn(SMLB_ENV, benchmark_name, dataset_dir, output_dir,
                        bench_args_list)

    # runtime output
    smlb_out = RuntimeOut(smlb_in.output_dir,
                          monitor_on=monitor_on,
                          monitor_interval=monitor_interval,
                          monitor_report_style=monitor_report_style)

    # create instance and run
    bench_run = Benchmark.create_instance(benchmark_name)
    try:
        bench_run(smlb_in, smlb_out)
    except Exception as e:
        # kill system monitor thread
        smlb_out.system.abort()
        raise e

    # report monitor
    smlb_out.report()


@cli.command(help='Display system information.')
@click.option('--usage/--no_usage', default=True,
              help='Include current usage of CPU, memory and GPU. '
                   'Default: True.')
def sysinfo(usage):
    """ sciml_bench sysinfo """
    print(format_info(all_sys_info(usage)))


@cli.command(help='About SciML-Benchmarks.')
def about():
    """ sciml_bench about """
    with open(Path(__file__).parent / 'messages/about.txt', 'r') as file:
        about_info = file.read()
    about_info = about_info.replace(
        'ver xx'.rjust(len(VERSION) + 4), f'ver {VERSION}')
    print(about_info)


if __name__ == '__main__':
    cli(auto_envvar_prefix='SCIML_BENCH')
