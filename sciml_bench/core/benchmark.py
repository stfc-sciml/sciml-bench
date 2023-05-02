#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# benchmark.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark management
"""
from datetime import datetime
import importlib.util
from pathlib import Path
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.utils import csv_to_stripped_set, display_logo, print_bullet_list, check_command
from subprocess import PIPE, STDOUT, run
import sciml_bench.core.dataset as Dataset
import sys

def create_training_instance(benchmark_name, bench_group, return_none_on_except=True):
    """ Create a benchmark instance for training"""
    try:
        # validate module
        bench_dir = Path(__file__).parents[1] / 'benchmarks' / bench_group
        file = str(bench_dir / benchmark_name / benchmark_name) + '.py'
        spec = importlib.util.spec_from_file_location(benchmark_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
      
         # create an instance by returning the sciml_bench_training function
        return getattr(mod, 'sciml_bench_training')
    except Exception as e:
        return None 



def create_inference_instance(benchmark_name, bench_group, return_none_on_except=True):
    """ Create a benchmark instance for inference"""
    try:
        # validate module - but models cannot be verified at this stage
        bench_dir = Path(__file__).parents[1] / 'benchmarks' / bench_group
        file = str(bench_dir / benchmark_name / benchmark_name) + '.py'
        spec = importlib.util.spec_from_file_location(benchmark_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # create an instance by returning the sciml_bench_inference function
        return getattr(mod, 'sciml_bench_inference')
    except Exception as e:
        return None


def install_benchmark_dependencies(benchmark_list, prog_env: ProgramEnv):
    """ Install benchmark dependencies """
    # input to set
    benchmarks = set(filter(''.__ne__, benchmark_list.split(',')))

    # handle all
    if 'all' in [bench.lower() for bench in benchmarks]:
        if len(benchmarks) != 1:
            print('"all" is presented - ignoring rest of the entries')
        benchmarks = set(prog_env.benchmarks.keys())

    # check benchmarks
    if not benchmarks.issubset(set(prog_env.benchmarks.keys())):
        print('\n\n')
        print(' ===============================================================\n')
        print(f' ERROR!!! One or more benchmarks in the list')
        print(f'  {benchmarks}')
        print(f' are not part of the SciML-Bench. Possible benchmark candidates are:')
        print_bullet_list(list(prog_env.benchmarks.keys()), 2)
        print()
        print(f' Aborting installation.')
        print(' ===============================================================\n')
        return 


    #log location for logging ourputs and errors
    log_file_dir = prog_env.output_dir  / 'install_logs'
    log_file_dir.mkdir(parents=True, exist_ok=True)

    log_file = str(log_file_dir) + '/install_' + \
                                datetime.today().strftime('%Y%m%d_%H%M') + \
                                '.log'

    display_logo()
    print(f'\n Installing dependencies.')
    # now install the dependencies
    benchmarks = list(benchmarks)
    benchmarks = { k: prog_env.benchmarks[k] for k in benchmarks }
    regular_deps, horovod_deps = build_dependencies(benchmarks)
    succeeded, failed =  install_dependencies(regular_deps, horovod_deps, log_file)
    # print benchmarks and dependencies
    print('\n ==================== Installation Summary ====================')
    print(' Benchmarks selected:')
    [print(f'  - {bench}') for bench in benchmarks]
    print(' Dependencies installed:')
    [print(f'  - {dep}') for dep in succeeded]
    if len(failed) > 0:
        print(' Dependencies failed:')
        [print(f'  - {failed_dep}') for failed_dep in failed]
        print(f'\n Detailed errors are logged to\n   {log_file}')

    else:
        print(f'\n Detailed outputs are logged to\n   {log_file}')
    print(' ===============================================================\n')



def install_dependencies(regular_deps, horovod_deps, log_file):
    # install anything other than horovod
    succeeded = set()
    failed = set()
    dependencies_copy = regular_deps.copy()

    for dep in dependencies_copy:
        print(f'\t- Attempting to install {dep}', end='', flush=True)
        result = run(f'pip install --no-cache-dir {dep}', stdout=PIPE, stderr=STDOUT, universal_newlines=True,  shell=True)
        if result.returncode != 0:
            failed.add(dep)
            print(f'...ERROR')
        else:
            succeeded.add(dep)
            print(f'...DONE')


        with open(log_file, 'a') as f:
            header = f'[Installing {dep}]'
            f.write(f'\n\n{header}\n')
            f.write("="*len(header))
            f.write('\n')
            f.write(result.stdout)
            f.write('\n\n\n')

    # Now start off with horovod
    h_succ, h_failed = __install_horovod__(horovod_deps, log_file)

    succeeded = succeeded.union(h_succ)
    failed = failed.union(h_failed)
    return succeeded, failed


def build_dependencies(benchmarks):
    # verify each benchmark
    regular_dependencies = set()
    horovod_dependencies  = set()
    reverse_lookup = {}
    for _,v in benchmarks.items():
        bench_deps = csv_to_stripped_set(v,'dependencies')
        for dep in bench_deps:
            if 'horovod' in dep:
                horovod_dependencies.add(dep)
            else:
                regular_dependencies.add(dep)
    
    return regular_dependencies, horovod_dependencies


def __get_horovod_env_key__(key: str):
    horovod_dict = {
    'horovod.torch'     : 'HOROVOD_WITH_PYTORCH=1', 
    'horovod.tensorflow': 'HOROVOD_WITH_TENSORFLOW=1',
    'horovod.mxnet'     : 'HOROVOD_WITH_MXNET=1'
    }
    if key in horovod_dict.keys():
        return horovod_dict[key]
    return ''

    

def __install_horovod__(horovod_dependencies, log_file):
    
    h_failed = set()
    h_succeeded = set()
    failed_deps = set()

    # Check whether there are dependencies in the first place
    if len(horovod_dependencies) == 0:
        return h_succeeded, horovod_dependencies

    # Next cmake - without cmake, we can't move an inch.
    if check_command('cmake') is False:
        with open(log_file) as f:
            header = f'[Installing Horovod Dependencies]'
            f.write(f'\n\n{header}\n')
            f.write("="*len(header))
            f.write('\n')
            f.write('Horovod relies on cmake. Please install cmake first!')
            f.write('\n\n\n')
            print(f'\t- Attempting to install  Horovod ...ERROR')
        return  h_succeeded, horovod_dependencies


    # check if horovod has been installed
    try:
        __import__('horovod')
        horovod_installed = True
    except:
        horovod_installed = False

    # If installed, try importing them.
    # If any of the dependencies doesn't work 
    # mark those for reinstallation 

    if horovod_installed:
        for mod in horovod_dependencies: 
            try:
                __import__(mod)
                h_succeeded.add(mod)
            except:
                failed_deps.add(mod)

    if (not horovod_installed) or len(failed_deps) > 0: 
        # Either Horovod is missing or there were some failures!
        # Rebuild what is needed to be installed 
        if len(failed_deps) > 0:
            horovod_dependencies = failed_deps

        env_str = ''
        for dep in horovod_dependencies: 
            env_str = __get_horovod_env_key__(dep)
            cmd_str = f'{env_str} pip install --no-cache-dir horovod'
            print(f'\t- Attempting to install {dep}',  end='', flush=True)
            result = run(cmd_str, stdout=PIPE, stderr=STDOUT, universal_newlines=True,  shell=True)

            if result.returncode:
                h_failed.add(dep)
                print(f'...ERROR')
            else:
                h_succeeded.add(dep)
                print(f'...DONE')

            with open(log_file, 'a') as f:
                header = f'[Installing {dep}]'
                f.write(f'\n\n{header}\n')
                f.write("="*len(header))
                f.write('\n')
                f.write(result.stdout)
                f.write('\n\n\n')

    return h_succeeded, h_failed

def __get_runnable_status__(is_good_train, is_good_inference, all_datasets_available):
    if is_good_train and is_good_inference and all_datasets_available:
        str = "Runnable (Training & Inference)"
    elif is_good_train and all_datasets_available:
        str = "Runnable (Training)"
    elif is_good_inference and all_datasets_available:
        str = "Runnable (Inference)"
    else:
        str = "Not runnable"
    return str

def get_status(benchmark_names, ENV:ProgramEnv):
    """
    Return benchmark status as runnable/not-runnable
    """
    flag = False
    status_str = []
    if not isinstance(benchmark_names, list):
        benchmark_names = [benchmark_names]
        flag =True

    for benchmark_name in benchmark_names:
        bench_group = ENV.get_bench_group(benchmark_name)
        is_good_train = create_training_instance(benchmark_name, bench_group, True)
        is_good_inference = create_inference_instance(benchmark_name, bench_group, True)
        dep_datasets = ENV.get_bench_datasets(benchmark_name)
        all_datasets_available = True
        if dep_datasets is not None:
            for ds in dep_datasets:
                all_datasets_available = all_datasets_available and Dataset.get_status(ds, ENV) == 'Downloaded' 
        else:
            all_datasets_available = True
        status_str.append(__get_runnable_status__(is_good_train, is_good_inference, all_datasets_available))
    
    if flag == True:
        status_str = status_str[0]
    
    return status_str

def get_benchmark_dataset_links(benchmark_names: dict, ENV: ProgramEnv):
    """
    returns the list of datasets that benchmarks are associated to
    as a list. 
    """
    is_single_item = False
    deps_str = []
    if not isinstance(benchmark_names, list):
        benchmark_names = [benchmark_names]
        is_single_item =True

    for benchmark_name in benchmark_names:
        datasets = ENV.get_bench_datasets(benchmark_name)
        deps_str.append(datasets)
    
    if is_single_item == True:
        deps_str = deps_str[0]
    
    return deps_str
