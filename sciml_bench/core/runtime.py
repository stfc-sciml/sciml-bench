#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# runtime.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Runtime input and output
"""

from datetime import datetime
from pathlib import Path
from sciml_bench.core.program import ProgramEnv
from sciml_bench.core.utils import SafeDict

from contextlib import contextmanager
from sciml_bench.core.utils import MultiLevelLogger
from sciml_bench.core.system import SystemMonitor
from sciml_bench.core.system import save_sys_info, save_proc_info


class RuntimeIn:
    """
    Class for runtime input

    Useful components of an object smlb_in:
    * smlb_in.start_time: start time of running as UTC-datetime
    * smlb_in.dataset_dir: dataset directory
    * smlb_in.output_dir: output directory
    * smlb_in.bench_args: benchmark-specific arguments
    """

    def __init__(self, smlb_env: ProgramEnv,
                 benchmark_name, dataset_dir, output_dir, bench_args_list):
        # check registration
        assert benchmark_name in smlb_env.registered_benchmarks.keys(), \
            f'Benchmark {benchmark_name} is not registered.\n' \
            f'Registered benchmarks: ' \
            f'{list(smlb_env.registered_benchmarks.keys())}'

        # start time
        self.start_time = datetime.utcnow().isoformat() + 'Z'

        # data dir
        if dataset_dir == '':
            # default
            dataset_name = \
                smlb_env.registered_benchmarks[benchmark_name]['dataset']
            self.dataset_dir = smlb_env.dataset_root_dir / dataset_name
        else:
            self.dataset_dir = Path(dataset_dir).expanduser()
        # check data existence
        assert self.dataset_dir.exists(), \
            f'Dataset directory does not exist: {self.dataset_dir}' \
            f'\nDownload dataset first or pass correct --dataset_dir.'

        # output dir
        assert output_dir != '', \
            'Option --output_dir cannot be "" (empty string).'

        # user-specified
        if output_dir[0] == '@':
            # special convention to use default root
            self.output_dir = smlb_env.output_root_dir / benchmark_name / \
                              output_dir[1:]
        else:
            self.output_dir = Path(output_dir).expanduser()

        # this is thread-safe
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # benchmark-specific arguments
        self.bench_args = SafeDict({})
        for key, val in bench_args_list:
            self.bench_args[key] = val


class RuntimeOut:
    """
    Class for runtime output, including logging and system monitoring

    Useful components of an object smlb_out:
    * smlb_out.log.console: multi-level logger on root (rank=0)
    * smlb_out.log.host: multi-level logger on host (local_rank=0)
    * smlb_out.log.device: multi-level logger on device (rank=any)
    * smlb_out.system: a set of system monitors
    """

    class Loggers:
        """ Collection of loggers """

        def __init__(self):
            """ Initialize loggers without activation """
            self.console = MultiLevelLogger()
            self.host = MultiLevelLogger()
            self.device = MultiLevelLogger()

        def activate(self, log_dir, rank, local_rank, activate_host=False,
                     activate_device=False, console_on_screen=True):
            """ Activate loggers """
            # console
            if rank == 0:
                self.console.activate('smlb_log_console',
                                      log_dir / 'console.log',
                                      screen=console_on_screen)
            # host
            # NOTE: rank rather than host ID is used in filename
            if activate_host and local_rank == 0:
                self.host.activate(f'smlb_log_host{rank}',
                                   log_dir / f'host{rank}.log',
                                   screen=False)
            # device
            if activate_device:
                self.device.activate(f'smlb_log_device{rank}',
                                     log_dir / f'device{rank}.log',
                                     screen=False)

        def begin(self, proc_name: str):
            """
            Begin a sub-process

            :param proc_name: name of the sub-process
            """
            self.console.begin(proc_name)
            self.host.begin(proc_name)
            self.device.begin(proc_name)

        def ended(self, proc_name: str = ''):
            """
            End a sub-process

            :param proc_name: name of the sub-process
            """
            self.console.ended(proc_name)
            self.host.ended(proc_name)
            self.device.ended(proc_name)

        def message(self, what: str):
            """
            Log a message to the current sub-process

            :param what: message text to log
            """
            self.console.message(what)
            self.host.message(what)
            self.device.message(what)

        @contextmanager
        def subproc(self, proc_name: str):
            """
            Log a sub-process using with statement

            :param proc_name: name of the sub-process
            """
            self.begin(proc_name)
            try:
                yield self
            finally:
                self.ended(proc_name)

    def __init__(self, output_dir, monitor_on, monitor_interval,
                 monitor_report_style):
        # this is a duplication of RuntimeIn.output_dir, so hide it
        self._output_dir = output_dir

        # create loggers without activation
        self.log = RuntimeOut.Loggers()

        # system monitor
        self._monitor_on = monitor_on
        self._monitor_report_style = monitor_report_style
        self.system = SystemMonitor(monitor_interval)

        # activation
        self._activated = False

    def activate(self, rank, local_rank, activate_log_on_host=False,
                 activate_log_on_device=False, console_on_screen=True):
        """
        Activate loggers and monitor

        :param rank: global rank of device
        :param local_rank: local rank of device on host
        :param activate_log_on_host: activate log on host (default: False)
        :param activate_log_on_device: activate log on device (default: False)
        :param console_on_screen: sync console log on screen (default: True)
        """
        # create log directory
        log_dir = self._output_dir / 'sciml_bench_run_logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # loggers
        self.log.activate(log_dir, rank, local_rank,
                          activate_host=activate_log_on_host,
                          activate_device=activate_log_on_device,
                          console_on_screen=console_on_screen)

        # monitor
        if self._monitor_on:
            # output directory
            sys_dir = self._output_dir / 'sciml_bench_run_sys'
            info_dir = sys_dir / 'info'
            info_dir.mkdir(parents=True, exist_ok=True)
            history_dir = sys_dir / 'history'
            history_dir.mkdir(parents=True, exist_ok=True)
            # time-independent
            if local_rank == 0:
                save_sys_info(info_dir, rank, style=self._monitor_report_style)
            save_proc_info(info_dir, rank, style=self._monitor_report_style)
            # time-dependent
            self.system.activate(rank, local_rank)

        # activation
        self._activated = True

    def report(self):
        """ Report monitor """
        if self._activated and self._monitor_on:
            history_dir = self._output_dir / 'sciml_bench_run_sys/history'
            self.system.report(history_dir, style=self._monitor_report_style)
