
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

from click.core import Context
from click.formatting import HelpFormatter
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
        
    def format_help(self, ctx: Context, formatter: HelpFormatter) -> None:
        display_logo()
        super().format_help(ctx, formatter)



@click.group(cls=NaturalOrderGroup)
@click.version_option(
    VERSION,  "--version", message="\nSciML-Bench Version %(version)s.\n"\
              "Copyright © 2021 SciML, RAL, STFC, UK. All rights reserved.\n"
)
def cli():
    pass


#####################################################################
# Primary Help                                                      #
#####################################################################


###################
# List Command 
###################

@cli.command('list', help='List datasets, benchmarks and examples.')
@click.argument('scope', default='all',
                type=click.Choice(['all', 'summary', 'datasets', 'benchmarks', 'examples'])) 
@click.option('--verify', is_flag=True, default=False,
              help='\b\nVerify existence of datasets, and modules of benchmarks.\n'\
                  'Default: False.')
def cmd_list(scope, verify):
    """ sciml_bench list """
    # key width first
    width_data = len(max(list(ENV.datasets.keys()), key=len))
    width_ben = len(max(list(ENV.benchmarks.keys()), key=len))
    width = max(width_data, width_ben)

    display_logo()
    # list datasets 
    if scope == 'summary' or scope == 'datasets' or scope=='all':
        print(' List of Datasets:')
        for name, props in ENV.datasets.items():
            if (props is None) or ('is_example' not in props) or ('is_example' in props and props['is_example'] == False):
                output = f'  * {name.ljust(width + 4)}'
                if verify: 
                    dataset_status =  Dataset.get_status(name, ENV)
                    output += f': {dataset_status}'
                print(output)
    
    # list benchmarks
    if scope == 'summary' or scope == 'benchmarks' or scope=='all':
        print('\n List of Benchmarks:')
        for name, props in ENV.benchmarks.items():
            if (props is None) or ('is_example' not in props) or ('is_example' in props and props['is_example'] == False):
                output = f'  * {name.ljust(width + 4)}'
                if verify: 
                    bench_status = Benchmark.get_status(name)
                    output += f': {bench_status}'
                print(output)
                
    # list example datasets and benchmarks
    if scope == 'examples' or scope=='all':
        n_ds_examples = sum(v["is_example"] == True for k, v in ENV.datasets.items() if v is not None and  'is_example' in v)
        n_bm_examples = sum(v["is_example"] == True for k, v in ENV.benchmarks.items() if v is not None and  'is_example' in v)

        if n_ds_examples > 0:
            print('\n List of Example Datasets:')
            for name, props in ENV.datasets.items():
                if props is not None and 'is_example' in props and props['is_example'] == True:
                    output = f'  * {name.ljust(width + 4)}'
                    if verify: 
                        dataset_status = Dataset.get_status(name, ENV)
                        output += f': {dataset_status}'
                    print(output)
        if n_bm_examples > 0:
            print('\n List of Example Benchmarks:')
            for name, props in ENV.benchmarks.items():
                if props is not None and 'is_example' in props and props['is_example'] == True:
                    output = f'  * {name.ljust(width + 4)}'
                    if verify: 
                        bench_status = Benchmark.get_status(name)
                        output += f': {bench_status}'
                    print(output)
    print('\n')


###################
# Info Command 
###################

@cli.command(help='Obtain a detailed information about an entity.')
@click.argument('entity')
def info(entity):
    """ sciml_bench info
    """
    
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

###################
# Install Command 
###################


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
                   'Default: dataset_dir in config.yml.\n')
@click.option('--mode', default='foreground',
                type=click.Choice(['foreground', 'background']),
                help='\b\nSets the downloading to foreground or background mode.\n'
                   'Default: foreground\n'
                )
@click.argument('dataset_name')
def download(dataset_name, dataset_dir, mode):
    """ sciml_bench download """

    display_logo()

    print(f'Downloading the dataset {dataset_name}\n'\
            f'in {mode} mode.\n') 

    dataset_dir = Dataset.download(dataset_name, Path(dataset_dir), ENV, mode)
    if dataset_dir is None:
        print('Download Failed.')
        return  
    if mode == 'background':
        print(f'A log is available at\n'\
              f'    {ENV.output_dir}/download_logs/.\n')  
    else: 
        print(f'\nDownload complete.  Downloaded/synced the dataset to\n'\
              f'    {dataset_dir}. \n')
    



###################
# Run Command 
###################

@cli.command(help='Run a given benchmark on a training/inference mode.')
@click.option('--mode', required=False, default = 'training',
              type=click.Choice(['training', 'inference']),
              help='\b\nSets the mode to training or inference.\n'
                   'Default: training.')
@click.option('--model', required=False,
              help='\b\nSets the model to be used (only for inference.)\n'
                   'Default: None.')                       
@click.option('--dataset_dir', required=False, 
              help='\b\nDirectory for the dataset(s).\n'
                   'Default: dataset directory from the config file')
@click.option('--output_dir', required=False,
              help='\b\nOutput directory for this run.\n'
                   'If not specified, outputs will be logged under\n' 
                   '        output_root_dir/benchmark_name/yyyymmdd/\n'
                   'where a yyyymmdd represents the current date\n'
                   'Use --output_dir=@foo to save outputs under\n'
                   '        output_root_dir/benchmark_name/foo/\n'
                   'If "@" is omitted, absolute path is assumed.\n')
@click.option('--monitor_on/--monitor_off', default=True,
              help='\b\nMonitor system usage during runtime.'
                   '\nDefault: Monitor On.')
@click.option('--monitor_interval', default=1.0, type=float,
              help='\b\nTime interval for system monitoring.'
                   '\nDefault: 1.0s.')
@click.option('--monitor_report_style', default='pretty',
              type=click.Choice(['pretty', 'yaml', 'hdf5']),
              help='\b\nReport style for system monitor.'
                   '\nDefault: Pretty.')
@click.option('--benchmark_specific', '-b', 'bench_args_list',
              type=(str, str), multiple=True,
              help='\b\nBenchmark-specific arguments.\n'
                   'Usage: -b key1 val1 -b key2 val2 ...')
@click.argument('benchmark_name')
def run(mode, model, dataset_dir, output_dir, monitor_on, 
        monitor_interval, monitor_report_style,
        bench_args_list, benchmark_name):
    """ sciml_bench run """

    # runtime input
    params_in = RuntimeIn(ENV, mode, model, benchmark_name, 
                          dataset_dir, output_dir, bench_args_list)

    if params_in.valid == False:
      print(params_in.error_msg)
      print()
      return

    # runtime output
    params_out = RuntimeOut(params_in.output_dir,
                          monitor_on=monitor_on,
                          monitor_interval=monitor_interval,
                          monitor_report_style=monitor_report_style)

    # create instance and run
    bench_types = ENV.get_bench_types(benchmark_name)
    if mode in bench_types:
        if mode ==  'inference':
            bench_run = Benchmark.create_inference_instance(benchmark_name)
        else:
            bench_run = Benchmark.create_training_instance(benchmark_name)
   
    if bench_run is None or mode not in bench_types:
        print(f'The benchmark {benchmark_name} does not support {mode}.')
        print(f'Terminating the execution')
        return 
    
    # Now try and launch
    try:    
        bench_run(params_in, params_out)
    except Exception as e:
        # kill system monitor thread
        params_out.system.abort()
        raise e
    

    # report monitor
    params_out.report()


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
    display_logo()
    print(about_info)


if __name__ == '__main__':
    cli(auto_envvar_prefix='SCIML_BENCH')
