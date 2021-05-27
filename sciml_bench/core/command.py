
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
import os
from pathlib import Path
import sciml_bench.core.benchmark as Benchmark
import sciml_bench.core.dataset as Dataset
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.system import all_sys_info, format_info
from sciml_bench import __version__ as VERSION
from sciml_bench.core.utils import display_logo
from sciml_bench.core.utils import extract_html_comments

# init a global ProgramEnv instance
ENV = ProgramEnv(Path(__file__).parents[1] / 'etc/configs/config.yml')


class NaturalOrderGroup(click.Group):
    """ Force click to keep the order of commands """
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=NaturalOrderGroup)
def cli():
    pass

###################
# List Command 
###################

@cli.command('list', help='List datasets, benchmarks and examples.')
@click.argument('scope', default='summary',
                type=click.Choice(['summary', 'datasets', 'benchmarks', 'examples']))
def cmd_list(scope):
    """ sciml_bench list """
    # key width first
    width_data = len(max(list(ENV.datasets.keys()), key=len))
    width_ben = len(max(list(ENV.benchmarks.keys()), key=len))
    width = max(width_data, width_ben)

    display_logo()
    # list datasets 
    if scope == 'summary' or scope == 'datasets':
        print('List of Datasets:')
        for name, props in ENV.datasets.items():
            if (props is None) or ('is_example' not in props) or ('is_example' in props and props['is_example'] == False):
                print(f'* {name.ljust(width + 4)}')
    
    # list benchmarks
    if scope == 'summary' or scope == 'benchmarks':
        print('\nList of Benchmarks:')
        for name, props in ENV.benchmarks.items():
            if (props is None) or ('is_example' not in props) or ('is_example' in props and props['is_example'] == False):
                print(f'* {name.ljust(width + 4)}')
                
    # list example datasets and benchmarks
    if scope == 'examples':
        n_ds_examples = sum(v["is_example"] == True for k, v in ENV.datasets.items() if v is not None and  'is_example' in v)
        n_bm_examples = sum(v["is_example"] == True for k, v in ENV.benchmarks.items() if v is not None and  'is_example' in v)

        if n_ds_examples > 0:
            print('List of Example Datasets')
            for name, props in ENV.datasets.items():
                if props is not None and 'is_example' in props and props['is_example'] == True:
                    print(f'* {name.ljust(width + 4)}')
        if n_bm_examples > 0:
            print('\nList of Example Benchmarks')
            for name, props in ENV.benchmarks.items():
                if props is not None and 'is_example' in props and props['is_example'] == True:
                    print(f'* {name.ljust(width + 4)}')
    print('\n')


###################
# Info Command 
###################

@cli.command('info', help='Provide a detailed information about a given dataset or benchmark')
@click.argument('entity')
def cmd_info(entity):
    """ sciml_bench info """
    # key width first
    width_data = len(max(list(ENV.datasets.keys()), key=len))
    width_ben = len(max(list(ENV.benchmarks.keys()), key=len))
    width = max(width_data, width_ben)

    # Decide whether the given entity is a benchmark or dataset:

    if entity in ENV.datasets.keys() or entity in ENV.benchmarks.keys():
      if entity in ENV.datasets.keys():
        info_path = Path(__file__).parents[1] / 'doc/datasets/' 
      else:
        info_path = Path(__file__).parents[1] / 'doc/benchmarks/' 
      content = extract_html_comments(str(info_path) + os.sep +  entity + '.md')
      if content:
        display_logo()
        print(f'{content.ljust(75)}')
        return

    print(f'No information can be found on the entity {entity}.\n')



@cli.command(help='Install benchmark dependencies.')
@click.argument('benchmark_list')
def install(benchmark_list):
    """ sciml_bench list """
    Benchmark.install_benchmark_dependencies(benchmark_list, ENV)


###################
# Download Command 
###################

@cli.command(help='Download a dataset.')
@click.option('--dataset_dir', default=ENV.dataset_dir,
              help='\b\nRoot directory of datasets.\n'
                   'Default: dataset_dir in config.yml.')
@click.option('--verify', is_flag=True,
              help='\b\nVerify the downloaded dataset(s).'
                   '\nDefault: False')
@click.argument('dataset_name')
def download(dataset_name, dataset_dir, verify):
    """ sciml_bench download """
    dataset_dir = Dataset.download(dataset_name, Path(dataset_dir), ENV)
    if dataset_dir is not None:
      print(f'A command for downloading the dataset {dataset_name}\n'\
            f'has been invoked in background mode. The dataset will be\n'\
            f'downloaded to {dataset_dir}. \n'\
            f'A log is available at {ENV.output_dir}/download_logs/\n')
    else:
      print('Download Failed.')

    if verify:
      print('<Verification of the downloaded dataset(s) is to be implemented>')

###################
# Run Command 
###################

@cli.command(help='Run a benchmark.')
@click.option('--dataset_dir', required=False, 
              help='\b\nDirectory of dataset.\n'
                   'Default: dataset directory from the config file')
@click.option('--output_dir', required=False,
              help='\b\nOutput directory of this run.\n'
                   'Convention: use --output_dir=@foo to save outputs under\n'
                   '            output_root_dir/benchmark_name/foo/;\n'
                   '            without "@", foo is used as a normal path.\n'
                   'If omitted, a new directory (based on yyyymmdd format)\n'
                   'inside the default path will be used.')
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
    smlb_in = RuntimeIn(ENV, benchmark_name, dataset_dir, output_dir,
                        bench_args_list)

    if smlb_in.valid == False:
      print(smlb_in.error_msg)
      print()
      return

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


###################
# Sysinfo Command 
###################


@cli.command(help='Display system information.')
def sysinfo():
    """ sciml_bench sysinfo """
    print(format_info(all_sys_info()))

###################
# About Command 
###################

@cli.command(help='About SciML-Benchmarks.')
def about():
    """ sciml_bench about """
    with open(Path(__file__).parents[1] / 'etc/messages/about.txt', 'r') as file:
        about_info = file.read()
    about_info = about_info.replace(
        'ver xx'.rjust(len(VERSION) + 4), f'ver {VERSION}')
    print(about_info)


if __name__ == '__main__':
    cli(auto_envvar_prefix='SCIML_BENCH')
