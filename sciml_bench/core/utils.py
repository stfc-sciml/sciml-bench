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
from sciml_bench import __version__ as VERSION
from bs4 import BeautifulSoup, Comment
from subprocess import PIPE, run

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


def display_logo():
    """ sciml_bench logo """
    fqfn = Path(__file__).parents[1] / 'etc/messages/logo.txt'
    with open(fqfn, 'r') as file:
        logo = file.read()
    logo = logo.replace('ver xx'.rjust(len(VERSION) + 4), f'ver {VERSION}')
    print(logo)



def extract_html_comments(file_name):
    """
    Extracts comments from an HTML File. Here This is used to extract 
    reportable short summary from markdown files.

    :param file_name: Fully qualified name of the MD file. 
    """
    try:
        with open(file_name, 'r') as source:
            html = source.read()
            soup = BeautifulSoup(html, 'lxml')
            comments = soup.findAll(text=lambda text:isinstance(text, Comment))
            if len(comments) > 0:
                return comments[0]
            else:
                return None
    except EnvironmentError:
        return None


def check_command(command_name):
    result = run(command_name, stderr=PIPE, stdout=PIPE, shell=True)
    if 'not found' in str(result.stderr):
        return False

    return True

def csv_to_stripped_set(dict, key):
    """
    Given a dictionary, and a key, returns 
    the list of items under that key as a set. 
    The white spaces are stripped 
    """
    if None in [dict, key]:
        return set()

    result = [x.strip() for x in dict[key].split(',')]
    return set(filter(''.__ne__, result))

def csv_string_to_stripped_set(string):
    """
    Given a csv string, returns 
    the list of items as a set. 
    The white spaces are stripped 
    """
    if string is None:
        return set()
    
    result = [x.strip() for x in string.split(',')]
    return set(filter(''.__ne__, result))


def print_items(heading, column1, column2=[]):
    """
    Given two columns of strings  (each as a list)
    prints them next to each other. 
    """

    if None in [heading, column1]:
        return 
    
    if column2 is None:
        column2 = []
        
    lengths = [len(x) for x in [column1, column2]]
    
    if lengths[0] ==0 or max(lengths)==0 or (lengths[1] > lengths[0]):
        return 
    
    max_length = len(max(column1, key=len)) +1
    print(f' {heading}\n ', end='')
    print('-'*len(heading))
    print()
    for i in range(len(column1)):
        flag_2 = i >= 0 and i < len(column2)
        output = f'  * {column1[i].ljust(max_length)}'
        if flag_2:
            output += f': {column2[i]}' 
        print(output)
    print()

    def list_all_files_in_dir(dataset_dir: Path):
        p = dataset_dir.glob('**/*')
        files = [x for x in p if x.is_file()]
        return sorted(files)