#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# config.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Configuration for the whole sciml-bench, including
* program configuration, and 
* datasets and benchmarks.
"""

from sciml_bench.core.utils import csv_to_stripped_set
from sciml_bench.core.utils import csv_string_to_stripped_set
import yaml
from pathlib import Path
from collections import defaultdict


class ProgramEnv:
    """ Class to initialize and store program environment """

    def __init__(self, config_file_path):
        # --------------
        # program config
        # --------------
        # read config
        with open(config_file_path) as handle:
            cfg = yaml.load(handle, yaml.SafeLoader)
            cfg = {} if cfg is None else cfg

        # Data Mirrors first
        self.mirrors = cfg['data_mirrors']
        # TODO: Workout the closest mirror 


        #Global download command
        self.download_commands = cfg['download_commands']
        
        # dirs
        cfg_dirs = cfg['directories']
        self.dataset_dir = Path(cfg_dirs['dataset_root_dir']).expanduser()
        self.output_dir  = Path(cfg_dirs['output_root_dir']).expanduser()
        self.model_dir   = Path(cfg_dirs['models_dir']).expanduser()
        self.docs_dir    = Path(cfg_dirs['docs_dir']).expanduser()

        # datasets 
        self.datasets = cfg['datasets']

        # benchmarks  
        self.benchmarks = cfg['benchmarks']
        self.benchmark_groups = defaultdict(list)
        for k, v in self.benchmarks.items():
            group_name = v['group']
            self.benchmark_groups[group_name].append(k)

        # Now validate the file 
        self.is_valid = False
        self.config_error = None
        self.__validate_config()


    def __validate_config(self):
        """
        Validates the configuration - minimum check
        """
        # At least one mirror
        if self.mirrors == None:
            self.is_valid = False
            self.config_error = 'Missing data mirrors.'
            return

        # At least one dataset
        if self.datasets == None:
            self.is_valid = False
            self.config_error = 'Missing datasets.'
            return

        # At least one benchmark
        if self.benchmarks == None:
            self.is_valid = False
            self.config_error = 'Missing benchmarks.'
            return

        # Check for the existence of download_command
        if self.download_commands == None:
            self.is_valid = False
            self.config_error = 'Missing download-commands.'
            return

        # Check for minimum records in datasets
        # end_points and sensible download_command
        for k, v in self.datasets.items():
            if 'end_point' not in v:
                self.is_valid = False
                self.config_error = 'Missing end-point for at least one dataset.'
                return   
            if 'download_command' in v and v['download_command'] not in self.download_commands:
                self.is_valid = False
                self.config_error = 'Invalid download command for at least one dataset.'
                return   

        self.is_valid = True

    def is_config_valid(self):
        return self.is_valid, self.config_error

    def get_download_command(self, dataset_name):
        """
        Extract the download command for a given dataset
        """
        cmd = None
        if self.is_valid == True:
            cmd = self.datasets[dataset_name]['download_command']
            cmd = self.download_commands[cmd]
        return cmd 

    # Given a benchmark, returns the sections of the benchmark
    # assigning default values wherever possible.
    def get_bench_sections(self, benchmark_name):
        """
        Extract various sections of a given benchmark
        """
        bench_datasets = self.get_bench_datasets(benchmark_name)
        bench_dependencies = self.get_bench_dependencies(benchmark_name)
        bench_types    = self.get_bench_types(benchmark_name)
        bench_group   = self.get_bench_group(benchmark_name)
        
        return bench_datasets, bench_dependencies, bench_types, bench_group


    def get_bench_types(self, benchmark_name):
        """
        Returns the benchmark type (inference or training or both) for a given benchmark name
        """
        if (benchmark_name not in self.benchmarks) or self.is_config_valid==False:
            return None

        benchmark = self.benchmarks[benchmark_name]
        if 'types' in benchmark.keys():
            return list(csv_to_stripped_set(benchmark, 'types'))
        else:
            return  ['training', 'inference']

    def get_bench_group(self, benchmark_name):
        """Get the group to which a benchmark belongs to"""
        if (benchmark_name not in self.benchmarks) or self.is_config_valid==False:
            return None

        benchmark = self.benchmarks[benchmark_name]
        if 'group' in benchmark.keys():
            return benchmark['group']
        else:
            return  ''

    def get_bench_dependencies(self, benchmark_name):
        """
        Return the package dependencies for a given benchmark
        """
        if (benchmark_name not in self.benchmarks) or self.is_config_valid==False:
            return None

        benchmark = self.benchmarks[benchmark_name]
        if 'dependencies' in benchmark.keys():
            return list(csv_to_stripped_set(benchmark, 'dependencies'))
        else:
            return  None

    def get_bench_datasets(self, benchmark_name):
        """
        Return the dataset dependencies for a given benchmark
        """
        if (benchmark_name not in self.benchmarks) or self.is_config_valid==False:
            return None

        benchmark = self.benchmarks[benchmark_name]
        if 'datasets' in benchmark.keys():
            return list(csv_to_stripped_set(benchmark, 'datasets'))
        else:
            return  None

    def list_benchmarks(self, group=None):
        """
        Returns a list of benchmarks for a given group.
        If the group is none or non-existent, return all as a list.
        """
        if (group is None) or (group not in self.benchmark_groups.keys()):
            output = [item for sublist in self.benchmark_groups.values() for item in sublist]
        else:
            output = self.benchmark_groups[group] 
        return output

    def list_datasets(self):
        """
        Returns a list of datasets as a list
        """
        output = []
        for name, props in self.datasets.items():
            output.append(name)
        return output  
