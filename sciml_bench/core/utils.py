#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# utils.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Smart tool collection
"""

import time
import logging
from pathlib import Path
from contextlib import contextmanager
import sys


class SafeDict(dict):
    """
    This derived class of dict enables checking key existence and type
    cast when getting a value. Return default if missing and raise
    ValueError if type cast fails.
    """

    def try_get(self, key, default):
        """ example: epochs = bench_args.try_get('epochs', 10) """
        # missing
        if key not in self.keys():
            return default
        # bool must be handled properly
        if isinstance(default, bool):
            if self[key].lower() in ['true', 't', 'yes', 'y', '1']:
                return True
            elif self[key].lower() in ['false', 'f', 'no', 'n', '0']:
                return False
            else:
                raise ValueError('Error casting a benchmark-specific argument:'
                                 f'\n* KEY   = {key};'
                                 f'\n* VALUE = {self[key]}'
                                 f'\n* TYPE  = {type(default).__name__}')
        # other types
        try:
            return type(default)(self[key])
        except:
            raise ValueError('Error casting a benchmark-specific argument:'
                             f'\n* KEY   = {key};'
                             f'\n* VALUE = {self[key]}'
                             f'\n* TYPE  = {type(default).__name__}')

    def try_get_dict(self, default_args):
        """ example: args = bench_args.try_get(default_args) """
        args = default_args.copy()
        for key, default in default_args.items():
            args[key] = self.try_get(key, default)
        return args


class MultiLevelLogger:
    """ Class to log multiple levels of sub-processes """

    class SimpleTimer:
        """ Simple timer class """

        def __init__(self, proc_name):
            self.start_time = time.time()
            self.proc_name = proc_name

        def elapsed(self):
            return time.time() - self.start_time

    def __init__(self, indent_char='.', indent_width=4):
        """
        Create a multi-level logger

        :param indent_char: character for indent fill (default: '.')
        :param indent_width: indent width (default: 4)
        """
        # indent
        self._indent_char = indent_char
        self._indent_width = indent_width

        # to be initialized later in activate() because
        # log activation is rank-dependent
        self._timers = []
        self._logger = None

    def activate(self, name, file_path, screen=True):
        """
        Activate this logger

        :param name: name of the logger
        :param file_path: file to save logs
        :param screen: show logs on screen (default: True)
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(
            logging.FileHandler(Path(file_path).expanduser(), 'w'))
        if screen:
            self._logger.addHandler(logging.StreamHandler())

    @property
    def activated(self):
        return self._logger is not None

    @property
    def current_level(self):
        return len(self._timers)

    @property
    def elapsed_shallowest(self):
        """ Elapsed time of the shallowest level """
        if self.current_level == 0:
            return 0.  # not activated or not called start()
        else:
            return self._timers[-1].elapsed()

    def begin(self, proc_name: str):
        """
        Begin a sub-process

        :param proc_name: name of the sub-process
        """
        if not self.activated:
            return

        # message
        message = [self._indent_char * self._indent_width * self.current_level,
                   '<BEGIN> ', proc_name]
        self._logger.info(''.join(message))
        # timer
        self._timers.append(MultiLevelLogger.SimpleTimer(proc_name))

    def ended(self, proc_name: str = ''):
        """
        End a sub-process

        :param proc_name: name of the sub-process
        """
        if not self.activated:
            return

        # timer
        timer = self._timers.pop()
        elapsed = timer.elapsed()
        # message
        if proc_name == '':
            # calling ended() without proc_name
            proc_name = timer.proc_name
        else:
            # calling ended() with proc_name, check consistency
            assert proc_name == timer.proc_name, \
                f"Subprocess names do not match in " \
                f"begin() and ended() of a MultiLevelLogger instance." \
                f"\nSubprocess name passed to begin(): {timer.proc_name}" \
                f"\nSubprocess name passed to ended(): {proc_name}"
        message = [self._indent_char * self._indent_width * self.current_level,
                   '<ENDED> ', proc_name, f' [ELAPSED = {elapsed:f} sec]']
        self._logger.info(''.join(message))

    def message(self, what: str):
        """
        Log a message to the current sub-process

        :param what: message text to log
        """
        if not self.activated:
            return

        # handle indentation of multiple lines
        for i, line in enumerate(what.split('\n')):
            if i == 0:
                message = [self._indent_char * self._indent_width *
                           self.current_level, '<MESSG> ', line.strip()]
            else:
                message = [self._indent_char * self._indent_width *
                           self.current_level, ' '.rjust(8, self._indent_char),
                           line.strip()]
            self._logger.info(''.join(message))

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


def query_yes_no(question, default=None):
    """
    Ask a yes/no question via input() and return the answer.

    :param question: a string presented to user
    :param default: the presumed answer if the user just hits <Enter>
    :return: True for "yes/y" or False for "no/n"
    """
    valid = {'yes': True, 'y': True,
             'no': False, 'n': False}
    if default is None:
        prompt = ' [yes/no] '
    elif default == 'yes':
        prompt = ' [YES/no] '
    elif default == 'no':
        prompt = ' [yes/NO] '
    else:
        raise ValueError(f'Invalid default answer {default}')
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
