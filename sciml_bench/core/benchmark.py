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

import importlib.util
from pathlib import Path
import os
from sciml_bench.core.program import ProgramEnv
from sciml_bench.core.utils import query_yes_no


def create_instance(benchmark_name, return_none_on_except=False):
    """ Create a benchmark instance """
    try:
        # validate module
        bench_dir = Path(__file__).parents[1] / 'benchmarks'
        file = str(bench_dir / benchmark_name / benchmark_name) + '.py'
        spec = importlib.util.spec_from_file_location(benchmark_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # create an instance by returning the sciml_bench_run function
        return getattr(mod, 'sciml_bench_run')
    except Exception as e:
        if return_none_on_except:
            return None
        else:
            raise e


def install_benchmark_dependencies(benchmark_list, smlb_env: ProgramEnv):
    """ Install benchmark dependencies """
    # input to set
    benchmarks = set(filter(''.__ne__, benchmark_list.split(',')))

    # handle all
    if 'all' in [bench.lower() for bench in benchmarks]:
        assert len(benchmarks) == 1, \
            'If "all" is presented, it must be the only entity.'
        benchmarks = set(smlb_env.registered_benchmarks.keys())

    # check benchmarks
    assert benchmarks <= smlb_env.registered_benchmarks.keys(), \
        f'Selected benchmarks must be in the registered set:' \
        f'\n{list(smlb_env.registered_benchmarks.keys())}'

    # verify each benchmark
    dependencies = set()
    horovod_env = {}
    for bench in benchmarks:
        if bench == 'MNIST_tf_keras':
            dependencies.add('tensorflow')
        elif bench == 'MNIST_torch':
            dependencies.add('torch')
            horovod_env['HOROVOD_WITH_PYTORCH'] = '1'
        elif bench == 'em_denoise':
            dependencies.add('mxnet-cu102')
        elif bench == 'dms_structure':
            dependencies.add('torch')
        elif bench == 'slstr_cloud':
            dependencies.add('tensorflow')
            dependencies.add('scikit-learn')
            horovod_env['HOROVOD_WITH_TENSORFLOW'] = '1'
        else:
            pass  # impossible

    # install anything other than horovod
    dependencies_copy = dependencies.copy()
    for dep in dependencies_copy:
        res = os.system(f'pip install {dep}')
        # installing mxnet-cuXXX will fail on MacOS; install no GPU version
        if 'mxnet-cu' in dep and res != 0:
            dep = 'mxnet'
            res = os.system(f'pip install {dep}')
            # change original for verbose
            dependencies.remove(dep)
            dependencies.add('mxnet')
        assert res == 0, f'"sciml-bench install" fails when doing' \
                         f'\n\n$ pip install {dep}\n'

    # install horovod
    if len(horovod_env) == 0:
        horovod_msg = "Horovod is not required by selected benchmarks."
    else:
        # message path
        msg_path = Path(__file__).parent / 'messages'

        # check if horovod has been installed
        try:
            __import__('horovod')
            horovod_installed = True
        except:
            horovod_installed = False

        if not horovod_installed:
            # command line
            env_str = ''
            for key, val in horovod_env.items():
                env_str += f'{key}={val} '
            cmd_str = f'{env_str}pip install horovod --no-cache-dir'

            # interactive session
            with open(msg_path / 'horovod_not_installed.txt', 'r') as file:
                msg = file.read()
            print(msg.replace('HOROVOD_CMD_STR', cmd_str))
            yes = query_yes_no('Install Horovod with the suggested '
                               'command line?', None)
            if yes:
                res = os.system(cmd_str)
                assert res == 0, f'"sciml-bench install" fails when doing' \
                                 f'\n\n$ {cmd_str}\n'
                horovod_msg = f'Horovod has been installed with' \
                              f'\n\n$ {cmd_str}\n'
            else:
                horovod_msg = f'Horovod is required and you chose to ' \
                              f'install it manually.\n' \
                              f'Suggested command line:\n\n$ {cmd_str}\n'
        else:
            # check if the current build satisfies new requirements
            horovod_modules = {
                'HOROVOD_WITH_PYTORCH': 'horovod.torch',
                'HOROVOD_WITH_TENSORFLOW': 'horovod.tensorflow',
                'HOROVOD_WITH_MXNET': 'horovod.mxnet'
            }
            try:
                for key, mod in horovod_modules.items():
                    if key in horovod_env.keys():
                        __import__(mod)
                requirements_satisfied = True
            except:
                requirements_satisfied = False
            # check satisfied
            if requirements_satisfied:
                horovod_msg = 'Horovod satisfies all requirements ' \
                              'from selected benchmarks.'
            else:
                # update cmd with already supported ML libraries
                for key, mod in horovod_modules.items():
                    try:
                        __import__(mod)
                        horovod_env[key] = '1'
                    except:
                        pass
                # command line
                env_str = ''
                for key, val in horovod_env.items():
                    env_str += f'{key}={val} '
                cmd_str = f'{env_str}pip install horovod --no-cache-dir'

                # interactive session
                with open(msg_path / 'horovod_installed.txt', 'r') as file:
                    msg = file.read()
                print(msg.replace('HOROVOD_CMD_STR', cmd_str))
                yes = query_yes_no('Re-install Horovod with the suggested '
                                   'command line?', None)
                if yes:
                    os.system('pip uninstall horovod --yes')
                    res = os.system(cmd_str)
                    assert res == 0, f'"sciml-bench install" fails when ' \
                                     f'doing\n\n$ {cmd_str}\n'
                    horovod_msg = f'Horovod has been re-installed with' \
                                  f'\n\n$ {cmd_str}\n'
                else:
                    horovod_msg = \
                        f'Horovod does not satisfy all the requirements and ' \
                        f'you chose to\ninstall it manually. ' \
                        f'Suggested command line:\n\n$ {cmd_str}\n'

    # print benchmarks and dependencies
    print('\n==================== Installation Finished ====================')
    print('Benchmarks selected:')
    [print(f'  - {bench}') for bench in benchmarks]
    print('Dependencies installed:')
    [print(f'  - {dep}') for dep in dependencies]
    print(f"\n{horovod_msg}")
    print('===============================================================\n')
