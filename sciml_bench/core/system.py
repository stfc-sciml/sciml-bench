#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# system.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
System info and monitoring
"""

import platform
import cpuinfo
import psutil
import socket
import GPUtil

import time
import yaml
import h5py
import numpy as np
from tabulate import tabulate
from threading import Timer
from subprocess import Popen, PIPE


def format_bytes(n_bytes, suffix='B'):
    """
    Scale bytes to a readable format
    e.g:
        1253656 => '1.20 MB'
        1253656678 => '1.17 GB'
    """
    factor = 1024
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E']:
        if n_bytes < factor:
            return f'{n_bytes:.2f} {unit}{suffix}'
        n_bytes /= factor


def node_info():
    """ node info """
    uname = platform.uname()
    info = {'System': uname.system,
            'Node': uname.node,
            'Release': uname.release,
            'Version': uname.version,
            'Machine': uname.machine,
            'Processor': uname.processor,
            'Python version': platform.python_version()}
    try:
        # this may fail in some cases
        info['IP address'] = socket.gethostbyname(socket.gethostname())
    except:
        pass
    return info


def cpu_info(usage=True):
    """ cpu info """
    freq = psutil.cpu_freq()  # min/max should be the same
    info = {'Brand': cpuinfo.get_cpu_info()['brand_raw'],
            'Num physical cores': psutil.cpu_count(logical=False),
            'Num logical cores': psutil.cpu_count(logical=True),
            'Min frequency': f'{freq.min} MHz',
            'Max frequency': f'{freq.max} MHz'}
    if usage:
        freq_per_core = psutil.cpu_freq(percpu=True)
        pct_per_core = psutil.cpu_percent(interval=.1, percpu=True)
        # frequency
        info['Current frequency'] = f'{freq.current} MHz'
        if len(freq_per_core) == len(pct_per_core):
            # core-wise freq supported
            info['Core-wise frequency'] = {}
            for i, freq_core in enumerate(freq_per_core):
                info['Core-wise frequency'][f'Core{i}'] = \
                    f'{freq_core.current} MHz'
        # usage
        info['Current load'] = f'{psutil.cpu_percent(interval=.1):.2f}%'
        info['Core-wise load'] = {}
        for i, pct_core in enumerate(pct_per_core):
            info['Core-wise load'][f'Core{i}'] = f'{pct_core:.2f}%'
    return info


def mem_info(usage=True):
    """ memory info """
    info = {'Physical': {}, 'Swap': {}}
    # physical
    mem = psutil.virtual_memory()
    info['Physical'] = {'Total': format_bytes(mem.total)}
    if usage:
        info['Physical'].update({'Available': format_bytes(mem.available),
                                 'Used': format_bytes(mem.used),
                                 'Percent': f'{mem.percent:.2f}%'})
    # swap
    swap = psutil.swap_memory()
    info['Swap'] = {'Total': format_bytes(swap.total)}
    if usage:
        info['Swap'].update({'Free': format_bytes(swap.free),
                             'Used': format_bytes(swap.used),
                             'Percent': f'{swap.percent:.2f}%'})
    return info


def disk_info(usage=True):
    """ disk info """
    # partitions
    info_all = {'Num partitions': 0}  # in case of no partition
    for i, partition in enumerate(psutil.disk_partitions()):
        try:
            disk_usage = psutil.disk_usage(partition.mountpoint)
        except:
            # this can happen if the disk isn't ready
            continue
        info = {'Device': partition.device,
                'Mount point': partition.mountpoint,
                'File system': partition.fstype,
                'Total': format_bytes(disk_usage.total)}
        if usage:
            info.update({'Free': format_bytes(disk_usage.free),
                         'Used': format_bytes(disk_usage.used),
                         'Percent': f'{disk_usage.percent:.2f}%'})
        info_all[f'Partition{i}'] = info
    info_all['Num partitions'] = len(info_all) - 1

    # IO stat (all physical disks)
    if usage:
        disk_io = psutil.disk_io_counters()
        info_all['IO since boot'] = {
            'Bytes read': format_bytes(disk_io.read_bytes),
            'Bytes written': format_bytes(disk_io.write_bytes)}
    return info_all


def gpu_info(usage=True):
    """ gpu info """
    info_all = {'Num GPUs': 0}  # in case of no GPU
    for i, gpu in enumerate(GPUtil.getGPUs()):
        info = {'ID': gpu.id,
                'UUID': gpu.uuid,
                'Name': gpu.name,
                'Serial': gpu.serial,
                'Total memory': gpu.memoryTotal}
        if usage:
            info.update({'Free memory': f'{gpu.memoryFree} MB',
                         'Used memory': f'{gpu.memoryUsed} MB',
                         'Current load': f'{gpu.load * 100:.2f}%',
                         'Temperature': f'{gpu.temperature}°C'})
        info_all[f'GPU{i}'] = info
    info_all['Num GPUs'] = len(info_all) - 1
    return info_all


def all_sys_info(usage=False):
    """ Get all system info """
    return {'Node': node_info(),
            'CPU': cpu_info(usage),
            'Memory': mem_info(usage),
            'Disk': disk_info(usage),
            'GPU': gpu_info(usage)}


def proc_info():
    """ process info """

    # memory transform
    def memory_transform(mem_obj):
        mem_dict_new = {}
        for key, n_bytes in mem_obj._asdict().items():
            mem_dict_new[key.upper()] = format_bytes(n_bytes)
        return mem_dict_new

    # map of useful stats
    # (old key, new key, value transform)
    useful_stats = [('pid', 'PID', lambda v: v),
                    ('name', 'Name', lambda v: v),
                    ('username', 'Username', lambda v: v),
                    ('cmdline', 'Cmdline', lambda v: ' '.join(v)),
                    ('cpu_num', 'Num CPUs', lambda v: v),
                    ('cpu_percent', 'CPU load', lambda v: f'{v:.2f}%'),
                    ('num_threads', 'Num Threads', lambda v: v),
                    ('memory_info', 'Memory', memory_transform),
                    ('memory_full_info', 'Memory', memory_transform),
                    ('memory_percent', 'Memory percent', lambda v: f'{v:.2f}%'),
                    ('io_counters', 'Disk IO',
                     lambda v: {'Bytes read': format_bytes(v.read_bytes),
                                'Bytes written': format_bytes(v.write_bytes)}),
                    ('environ', 'Environment variables', lambda v: v)]

    # get process and measure
    process = psutil.Process()
    proc_dict = process.as_dict()

    # extract useful stats
    info = {}
    for (old_key, new_key, transform) in useful_stats:
        val = proc_dict.pop(old_key, None)
        if val is not None:
            info[new_key] = transform(val)

    # merge memory
    if 'Memory percent' in info.keys() and 'Memory' in info.keys():
        info['Memory']['Percent'] = info.pop('Memory percent')

    # re-measure cpu; otherwise load is always 0.0
    if 'CPU load' in info.keys():
        info['CPU load'] = f'{process.cpu_percent():.2f}%'

    # GPU? No GPU utilization when proc_info() is called
    return {'Process': info}


def format_info(info):
    """ Print info neatly """
    sec_width = 64
    eq = '    =    '
    # find key width
    key_widths = []
    for section, properties in info.items():
        for prop_key, prop_val in properties.items():
            if type(prop_val) is dict:
                key_widths.append(len(max(list(prop_val.keys()), key=len)) + 4)
            else:
                key_widths.append(len(prop_key))
    key_width = max(key_widths)
    # format items
    msg = []
    for section, properties in info.items():
        n0 = (sec_width - 2 - len(section)) // 2
        n1 = n0 if n0 * 2 + 2 + len(section) == sec_width else n0 + 1
        msg.append('\n' + '=' * n0 + f' {section} ' + '=' * n1)
        for prop_key, prop_val in properties.items():
            if type(prop_val) is dict:
                msg.append((prop_key + ' ').ljust(sec_width, '_'))
                for sub_key, sub_val in prop_val.items():
                    msg.append(' ' * 4 + sub_key.ljust(key_width - 4) +
                               eq + str(sub_val))
            else:
                msg.append(prop_key.ljust(key_width) + eq + str(prop_val))
        msg.append('=' * (n0 + n1 + 2 + len(section)))
    return '\n'.join(msg)


def save_records_yaml(path, records):
    """ save records to yaml """
    try:
        # cast np.ndarray to list to avoid messy output
        records['Time']['Event'] = records['Time']['Event'].tolist()
    except:
        pass
    with open(path, 'w') as handle:
        yaml.dump(records, handle)


def save_records_hdf5(path, records):
    """ save records to hdf5 """
    with h5py.File(path, 'w') as root:
        for section, properties in records.items():
            grp = root.create_group(section)
            for prop_key, prop_val in properties.items():
                if prop_key == 'Event':  # string-typed array
                    grp.create_dataset(
                        prop_key, data=np.array(prop_val),
                        dtype=h5py.special_dtype(vlen=str))
                else:
                    grp.create_dataset(prop_key, data=prop_val)


def save_info(info, output_dir, filename, style):
    """ Save info """
    if style == 'pretty':
        with open(output_dir / f'{filename}.txt', 'w') as handle:
            handle.write(format_info(info))
    else:
        # info does not contain any array; no need to use hdf5
        save_records_yaml(output_dir / f'{filename}.yml', info)


def save_sys_info(output_dir, rank, style, usage=True):
    """ Save system info """
    save_info(all_sys_info(usage), output_dir, f'machine_on_host{rank}', style)


def save_proc_info(output_dir, rank, style):
    """ Save process info """
    save_info(proc_info(), output_dir, f'process_on_device{rank}', style)


# --------------------
# System monitor class
# --------------------

# use MB as unit for bytes
mega = 1024 * 1024


def memory_transform_runtime(mem_obj):
    """ Transform memory usage """
    mem_dict_new = {}
    for key, n_bytes in mem_obj._asdict().items():
        mem_dict_new[f'{key.upper()} (MB)'] = n_bytes / mega
    return mem_dict_new


def gpu_mem_by_proc(pid):
    """ GPU memory used by a process """
    try:
        # --query-compute-apps does not support gpu_util
        # --query-accounted-apps is not for runtime
        p = Popen(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                   '--format=csv,noheader,nounits'], stdout=PIPE)
        stdout, stderr = p.communicate()
    except:
        return 0.
    # parse results
    mem = 0.
    lines = stdout.decode('UTF-8').splitlines()
    for line in lines:
        values = line.split(', ')
        if pid == int(values[0]):
            mem += float(values[1])
    return mem


class SystemMonitor:
    """ Class for system monitoring """

    class RepeatTimer(Timer):
        """ Repeated timer (threaded) """

        def run(self):
            while not self.finished.wait(self.interval):
                self.function(*self.args, **self.kwargs)

    # map of useful stats
    # {old key: (new key, value transform)}
    _useful_stats = {'cpu_num': ('NumCPUs', lambda v: v),
                     'cpu_percent': ('CPULoad (%)', lambda v: v),
                     'num_threads': ('NumThreads', lambda v: v),
                     'memory_info': ('Memory', memory_transform_runtime),
                     'memory_full_info': ('Memory', memory_transform_runtime),
                     'memory_percent': ('MemoryPercent (%)', lambda v: v),
                     'io_counters': ('IO', lambda v: {
                         'Read (MB)': v.read_bytes,
                         'Written (MB)': v.write_bytes})
                     }

    @staticmethod
    def init_records(host):
        """ Initialize records """
        # --------------
        # device records
        # --------------
        header = {
            'Time': ['Elapsed (sec)'],
            'Process': []}

        # get process and measure
        process = psutil.Process()
        proc_dict = process.as_dict()

        # extract useful stats
        valid_process_keys = []
        for old_key, (new_key, trans) in SystemMonitor._useful_stats.items():
            val = proc_dict.pop(old_key, None)
            if val is not None:
                if new_key in ['Memory', 'IO']:  # flatten sub-items
                    for sub_key, sub_val in trans(val).items():
                        header['Process'].append(new_key + sub_key)
                else:
                    header['Process'].append(new_key)
                valid_process_keys.append(old_key)
        # remove memory_info if memory_full_info exists
        if ('memory_full_info' in valid_process_keys and
                'memory_info' in valid_process_keys):
            valid_process_keys.remove('memory_info')

        # gpu
        header['Process'].append('GPUMemory (MB)')

        # ------------
        # host records
        # ------------
        if host:
            header.update({
                'CPU': [],
                'Memory': ['PhysTotal (MB)', 'PhysAvail (MB)',
                           'PhysUsed (MB)', 'PhysPercent (%)',
                           'SwapTotal (MB)', 'SwapFree (MB)',
                           'SwapUsed (MB)', 'SwapPercent (%)'],
                'Disk': ['Read (MB)', 'Written (MB)'],
                'GPU': []})

            # individual cpu
            freq_per_core = psutil.cpu_freq(percpu=True)
            pct_per_core = psutil.cpu_percent(percpu=True)
            header['CPU'].append(f'Frequency (MHz)')
            if len(freq_per_core) == len(pct_per_core):
                # to make output concise, add core-wise frequency only when
                # frequency varies by cores (most nodes have the same cores)
                freq_all = np.zeros(len(freq_per_core))
                for i, freq in enumerate(freq_per_core):
                    freq_all[i] = freq.current
                freq_mean = np.mean(freq_all)
                freq_max_var = np.max(np.abs(freq_all - freq_mean))
                if freq_max_var > freq_mean * .01:
                    for i, freq in enumerate(freq_per_core):
                        header['CPU'].append(f'Core{i}:Frequency (MHz)')
            header['CPU'].append(f'Usage (%)')
            for i, pct in enumerate(pct_per_core):
                header['CPU'].append(f'Core{i}:Usage (%)')

            # individual gpu
            for i, gpu in enumerate(GPUtil.getGPUs()):
                header['GPU'].append(f'GPU{i}:MemTotal (MB)')
                header['GPU'].append(f'GPU{i}:MemFree (MB)')
                header['GPU'].append(f'GPU{i}:MemUsed (MB)')
                header['GPU'].append(f'GPU{i}:Load (%)')
                header['GPU'].append(f'GPU{i}:Temperature (°C)')

        # create empty records based on the header
        records = {}
        for section, properties in header.items():
            records[section] = {}
            for prop_key in properties:
                records[section][prop_key] = []
        return records, process, valid_process_keys

    @staticmethod
    def append_to_records(records, start_time, host,
                          process, valid_process_keys, start_disk_read_write):
        """ Append current state to records """
        # time
        records['Time']['Elapsed (sec)'].append(time.time() - start_time)

        # --------------
        # device records
        # --------------
        # measure
        proc_dict = process.as_dict()

        # extract useful stats
        for old_key in valid_process_keys:
            new_key = SystemMonitor._useful_stats[old_key][0]
            transform = SystemMonitor._useful_stats[old_key][1]
            # perform value transform
            val = transform(proc_dict.pop(old_key, None))
            if type(val) is dict:
                for sub_key, sub_val in val.items():
                    records['Process'][new_key + sub_key].append(sub_val)
            else:
                records['Process'][new_key].append(val)

        # GPU
        mem = gpu_mem_by_proc(process.pid)
        records['Process']['GPUMemory (MB)'].append(mem)

        # ------------
        # host records
        # ------------
        if not host:
            return
        # cpu
        # frequency
        records['CPU']['Frequency (MHz)'].append(psutil.cpu_freq().current)
        if 'Core0:Frequency (MHz)' in records['CPU'].keys():
            for i, freq in enumerate(psutil.cpu_freq(percpu=True)):
                records['CPU'][f'Core{i}:Frequency (MHz)'].append(freq.current)
        # usage
        # NOTE: cpu_percent() is called in init_records(), so no need to
        #       specify interval (reporting usage since last call)
        records['CPU']['Usage (%)'].append(psutil.cpu_percent())
        for i, pct in enumerate(psutil.cpu_percent(percpu=True)):
            records['CPU'][f'Core{i}:Usage (%)'].append(pct)

        # memory
        mem = psutil.virtual_memory()
        records['Memory']['PhysTotal (MB)'].append(mem.total / mega)
        records['Memory']['PhysAvail (MB)'].append(mem.available / mega)
        records['Memory']['PhysUsed (MB)'].append(mem.used / mega)
        records['Memory']['PhysPercent (%)'].append(mem.percent)
        swap = psutil.swap_memory()
        records['Memory']['SwapTotal (MB)'].append(swap.total / mega)
        records['Memory']['SwapFree (MB)'].append(swap.free / mega)
        records['Memory']['SwapUsed (MB)'].append(swap.used / mega)
        records['Memory']['SwapPercent (%)'].append(swap.percent)

        # disk
        disk_io = psutil.disk_io_counters()
        records['Disk']['Read (MB)'].append(
            (disk_io.read_bytes - start_disk_read_write[0]) / mega)
        records['Disk']['Written (MB)'].append(
            (disk_io.write_bytes - start_disk_read_write[1]) / mega)

        # gpu
        for i, gpu in enumerate(GPUtil.getGPUs()):
            records['GPU'][f'GPU{i}:MemTotal (MB)'].append(gpu.memoryTotal)
            records['GPU'][f'GPU{i}:MemFree (MB)'].append(gpu.memoryFree)
            records['GPU'][f'GPU{i}:MemUsed (MB)'].append(gpu.memoryUsed)
            records['GPU'][f'GPU{i}:Load (%)'].append(gpu.load * 100)
            records['GPU'][f'GPU{i}:Temperature (°C)'].append(gpu.temperature)

    def __init__(self, interval):
        """ Create monitor without activation (rank-dependent) """
        self._interval = interval
        self._rank = None
        self._host = None
        self._records = None
        self._start_time = None
        self._timer = None
        self._events = {}

    def activate(self, rank, local_rank):
        """ Activate monitor """
        self._rank = rank
        self._host = (local_rank == 0)
        # --------------
        # initialization
        # --------------
        # records
        self._records, process, valid_process_keys = \
            SystemMonitor.init_records(self._host)
        # initial IO
        disk_io = psutil.disk_io_counters()
        start_disk_read_write = [disk_io.read_bytes, disk_io.write_bytes]
        # clear events
        self._events = {}

        # ---------------
        # start threading
        # ---------------
        # wall-clock timer
        self._start_time = time.time()
        # repeated timer
        self._timer = SystemMonitor.RepeatTimer(
            self._interval, SystemMonitor.append_to_records,
            args=(self._records, self._start_time, self._host,
                  process, valid_process_keys, start_disk_read_write))
        self._timer.start()

    def stamp_event(self, event: str):
        """
        Add an event to the current timestamp
        :param event: event info
        """
        # not activated; must check because this is called by contributor
        if self._timer is None:
            return
        event_time = time.time() - self._start_time
        self._events[event_time] = event

    def report(self, output_dir, style):
        """ Stop monitor and report results """
        if self._timer is None:
            return

        # stop timer
        self._timer.cancel()

        # add events to records
        times = np.array(self._records['Time']['Elapsed (sec)'])
        self._records['Time']['Event'] = \
            np.full((len(times),), '-', dtype=object)
        for event_time, event in self._events.items():
            index = np.abs(times - event_time).argmin()
            if self._records['Time']['Event'][index] == '-':
                # first event
                self._records['Time']['Event'][index] = event
            else:
                # multiple events happened closely
                self._records['Time']['Event'][index] += f'; {event}'

        # save to file
        if style == 'pretty':
            # one file per section
            for section, properties in self._records.items():
                if section == 'Time':
                    continue
                # all files start with time column
                headers = ['Elapsed (sec)', ]
                data = [self._records['Time']['Elapsed (sec)'], ]
                fmt = ['.2f', ]
                # add property columns
                for prop_key, prop_val in properties.items():
                    headers.append(prop_key)
                    data.append(prop_val)
                    fmt.append('.6f' if 'MB' in prop_key else '.2f')
                # all files end with event column
                headers.append('Event')
                data.append(self._records['Time']['Event'])

                # truncate data to same size
                min_len = len(min(data, key=len))
                for i, row in enumerate(data):
                    data[i] = row[0:min_len]

                # write to file
                if section == 'Process':
                    # Process is on device
                    file = f'process_on_device{self._rank}.txt'
                else:
                    # all others are on host
                    file = f'machine_on_host{self._rank}_{section}.txt'
                with open(output_dir / file, 'w') as handle:
                    handle.write(tabulate(np.array(data, dtype=object).T,
                                          headers=headers, floatfmt=fmt))
        elif style == 'yaml':
            # all in two files
            device_records = self._records.pop('Process')
            save_records_yaml(output_dir / f'process_on_device{self._rank}.yml',
                              {'Time': self._records['Time'],
                               'Process': device_records})
            if self._host:
                save_records_yaml(output_dir /
                                  f'machine_on_host{self._rank}.yml',
                                  self._records)
        elif style == 'hdf5':
            # all in two files
            device_records = self._records.pop('Process')
            save_records_hdf5(output_dir / f'process_on_device{self._rank}.h5',
                              {'Time': self._records['Time'],
                               'Process': device_records})
            if self._host:
                save_records_hdf5(output_dir /
                                  f'machine_on_host{self._rank}.h5',
                                  self._records)
        else:
            pass  # impossible

    def abort(self):
        """ Abort if main thread fails """
        if self._timer is not None:
            self._timer.cancel()
