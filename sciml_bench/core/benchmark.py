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
import os
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.utils import display_logo, query_yes_no


def create_training_instance(benchmark_name, return_none_on_except=False):
    """ Create a benchmark instance for training"""
    try:
        # validate module
        bench_dir = Path(__file__).parents[1] / 'benchmarks'
        file = str(bench_dir / benchmark_name / benchmark_name) + '.py'
        spec = importlib.util.spec_from_file_location(benchmark_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # create an instance by returning the sciml_bench_training function
        return getattr(mod, 'sciml_bench_training')
    except Exception as e:
        if return_none_on_except:
            return None
        else:
            raise e



def create_inference_instance(benchmark_name, return_none_on_except=False):
    """ Create a benchmark instance for inference"""
    try:
        # validate module - but models cannot be verified at this stage
        bench_dir = Path(__file__).parents[1] / 'benchmarks'
        file = str(bench_dir / benchmark_name / benchmark_name) + '.py'
        spec = importlib.util.spec_from_file_location(benchmark_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # create an instance by returning the sciml_bench_inference function
        return getattr(mod, 'sciml_bench_inference')
    except Exception as e:
        if return_none_on_except:
            return None
        else:
            raise e


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
    if benchmarks <= prog_env.benchmarks.keys():
        print(f'One or more benchmarks in the list')
        print(f'* {benchmarks}')
        print(f'is not part of the SciML-Bench' )
        return 


    #log location for logging ourputs and errors
    log_file_dir = prog_env.output_dir  / 'install_logs'
    log_file_dir.mkdir(parents=True, exist_ok=True)

    log_file = str(log_file_dir) + '/install_' + \
                                datetime.today().strftime('%Y%m%d') + \
                                '.log'

    display_logo()
    print(f'\n Installing dependencies.')
    # now install the dependencies
    regular_deps, horovod_deps = build_dependencies(prog_env.benchmarks)
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
    print(f'\n Detailed Errors/Outputs are logged to\n   {log_file}')
    print(' ===============================================================\n')


def install_dependencies(regular_deps, horovod_deps, log_file):
    # install anything other than horovod
    succeeded = set()
    failed = set()
    dependencies_copy = regular_deps.copy()
    for dep in dependencies_copy:
        print(f'\t- Attempting to install {dep}', end='', flush=True)
        res = os.system(f'pip install {dep} &> {log_file}')
        if res != 0:
            failed.add(dep)
            print(f'...ERROR')
        else:
            succeeded.add(dep)
            print(f'...DONE')
    
    # Now start off with horovod
    h_succ, h_failed = __install_horovod__(horovod_deps, log_file)
    succeeded = succeeded.union(h_succ)
    failed = failed.union(h_failed)
    return succeeded, failed


def build_dependencies(benchmarks):
    # verify each benchmark
    regular_dependencies = set()
    horovod_dependencies  = set()
    for _,v in benchmarks.items():
        bench_deps = set(filter(''.__ne__, v['dependencies'].split(',')))
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
        return 

    # check if horovod has been installed
    try:
        __import__('horovod')
        horovod_installed = True
    except:
        horovod_installed = False

    # If installed, try importing them m
    # If any of the dependencies don't work 
    # mark those for reinstallation 

    if horovod_installed:
        for mod in horovod_dependencies: 
            try:
                __import__(mod)
                h_succeeded.add(mod)
            except:
                failed_deps.add(mod)

    if (not horovod_installed) or len(failed_deps) > 0: 
        # Either Horovod is missing or there were some failures
        # Rebuild what is needed to be installed 
        if len(failed_deps) > 0:
            horovod_dependencies = failed_deps

        env_str = ''
        for dep in horovod_dependencies: 
            env_str = __get_horovod_env_key__(dep)
            cmd_str = f'{env_str} pip install horovod --no-cache-dir &> {log_file}'
            print(f'\t- Attempting to install {dep}',  end='', flush=True)
            res = os.system(cmd_str)
            if res != 0:
                h_failed.add(dep)
                print(f'...ERROR')

            else:
                h_succeeded.add(dep)
                print(f'...DONE')



    return h_succeeded, h_failed
