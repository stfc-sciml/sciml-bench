#MIT License
#
#Copyright (c) 2016 Nouamane Laanait.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import json
import numpy as np
from collections import OrderedDict
import multiprocessing as mp
import sys
from os import listdir, path
import re
import os


def json_to_ordered_dict(file):
    """
    Reads a timeline.json file output by Tensorflow/libcupti and returns and OrderedDict object
    :param file: .json file.
    :return: OrderedDict
    """
    with open(file, mode='r') as f:
        def _as_ordered_dict(val):
            return OrderedDict(val)

        output = json.load(f, object_hook=_as_ordered_dict, object_pairs_hook=_as_ordered_dict)
        dic = OrderedDict(output)

    return dic


def get_all_ops(trace_dic):
    """
    Params:
    trace_dic: collections.OrderedDict of traceEvent
    Return: list of dictionaries of all ops.
    """
    try:
        trace_events = trace_dic['traceEvents']
    except KeyError:
        print('Not valid GPU trace dict object.')
        sys.exit()
    all_ops = []
    for trace in trace_events:
        try:
            if trace['cat'] == 'Op':
                all_ops.append(trace)
        except KeyError:
            pass
    return all_ops


def get_stream_all(trace_dic):
    """
    Params:
    trace_dic: collections.OrderedDict of traceEvent
    Return: pid of GPU/stream:all, (stream, pid) dictionary
    """
    try:
        trace_events = trace_dic['traceEvents']
    except KeyError:
        print('Not valid GPU trace dict object.')
        sys.exit()
    all_procs = []
    for trace in trace_events:
        try:
            if trace['name'] == 'process_name':
                all_procs.append((trace['args']['name'], trace['pid']))
        except KeyError:
            pass
    dic_procs = dict(all_procs)
    pid = dic_procs['/device:GPU:0/stream:all Compute']
    return dic_procs, pid


def get_unique_ops_names(all_ops):
    """
    Find unique op names.
    Params:
    all_ops: list, of dictionary of all operations.
    Return: list of unique op names.
    """
    return set(op['name'] for op in all_ops)


def get_wall_duration(op_names, all_ops, pid_list=(11, 7, 13, 15, 9)):
    """
    Calculates wall duration for each op in op_names.
    Params:
    op_names: list (str), names of ops of interest.
    pid_list: list (str), names of pid to include.
    all_ops: output of get_all_ops().
    Return:
    total wall duration, dict['op'] = wall duration.
    """
    # 1. Construct dictionary of op with name matching op_names
    ops_dic = OrderedDict()
    for name in op_names:
        ops = []
        for op in all_ops:
            if op['name'] == name:
                ops.append(op)
        ops_dic[name] = ops

    # 2. get duration for each op
    op_dict = OrderedDict()
    total_dur = 0
    for op_name in op_names:
        op_dur = 0
        for itm in ops_dic[op_name]:
            if itm['pid'] in pid_list:
                op_dur += itm['dur']
        op_dict[op_name] = op_dur * 1e-3  # convert from us to ms
        total_dur += op_dur * 1e-3

    # fixing the NCCL key:
    op_dict['unknown (nccl AllReduceKernel_sum_)'] = op_dict.pop('unknown')

    # Sorting durations:
    sorted_dur = sorted(op_dict.items(), key=lambda x: x[1])[::-1]
    # sorted_dur = sorted(op_dict.items(), key=operator.itemgetter(1))

    return OrderedDict(sorted_dur), total_dur


def print_timeline_stats(sorted_dur, total_dur, min_msec=5):
    """
    Prints the total time and times per op so long as the time was > min_msec
    :param sorted_dur: OrderedDict object with time per op. Times in msec
    :param total_dur: Number - total wall time per step. Time in msec
    :param min_msec: Number, optional - minimum wall time for op
    """
    print('Total Wall Duration (ms): %4.3f\n' % total_dur)
    print('OPS with wall duration > 5 ms:')
    for key, val in sorted_dur.items():
        if val > min_msec:
            print('%s : %3.3f ms' % (key, val))


def parse_single_timeline(curr_file):
    """
    Parses a single timeline file and extracts the time per op and total wall time

    :param curr_file: str / unicode - path to a single timeline .json file
    :return dicts: OrderedDict object with time per op. Times in msec
    :return tot_times: Number - total wall time per step. Time in msec
    """
    dic = json_to_ordered_dict(curr_file)
    all_ops = get_all_ops(dic)
    unique_op_names = get_unique_ops_names(all_ops)
    proc_dic, stream_all_pid = get_stream_all(dic)
    sorted_dur_dicts, total_dur = get_wall_duration(unique_op_names, all_ops, pid_list=[stream_all_pid])
    return sorted_dur_dicts, total_dur


def parse_all_timeline_files(folder, prefix='timeline', suffix='.json'):
    """
    Parses all timeline files in the given dictionary to extract the times per op and total wall time

    :param folder: str / unicode - path to directory containing all the timeline json files
    :param prefix: str / unicode (optional) - Prefix for the file names. Default = 'timeline'
    :param suffix: str / unicode (optional) - suffix for the file names. Default = '.json'
    :return dicts: list of OrderedDict objects per timeline file. Times in msec
    :return tot_times: list of Numbers with the total wall time per step. Times in msec
    """

    files = []
    for name in listdir(folder):
        if name.startswith(prefix) and name.endswith(suffix):
            files.append(path.join(path.abspath(folder), name))

    dicts = []
    tot_times = []
    if len(files) > 16:
        cores = 4
        pool = mp.Pool(cores)
        jobs = pool.imap(parse_single_timeline, files)
        results = [j for j in jobs]
        pool.close()
        for item in results:
            dicts.append(item[0])
            tot_times.append(item[1])
    else:
        for curr_file in files:
            sorted_dur_dicts, total_dur = parse_single_timeline(curr_file)
            dicts.append(sorted_dur_dicts)
            tot_times.append(total_dur)

    return dicts, tot_times


def parse_nvprof_csv(nvprof_csv):
    """
    Extract data from nvprof and calculate/return OPS, FLOPS.
    """
    p = re.compile(r'Device')
    with open(nvprof_csv) as f:
        skip_ln = 0
        while (True):
            line = f.readline()
            match = p.search(line)
            if match:
                fields = line
                skip_ln += 1
                break
            if skip_ln > 20:
                print('The provided file is missing nvprof headers!')
                break
            skip_ln += 1
    fields = fields.split(',')
    # Now that the number of header rows are known, the rest can be extracted easily
    arr = np.genfromtxt(nvprof_csv, skip_header=skip_ln, delimiter='Floating Point Operations(Single Precision)',
                        comments='==', dtype=None, encoding=None)

    logs = dict()
    # it would have been easier if we could use pandas dataframes but that's not available
    for lhs, rhs in arr:
        lhs_splits = lhs.split(',')
        rhs_splits = rhs.split(',')
        logs[','.join(lhs_splits[1:-3])] = {'invocations': int(lhs_splits[-3]),
                                            'min_ops': int(float(rhs_splits[1])),
                                            'max_ops': int(float(rhs_splits[2])),
                                            'avg_ops': int(float(rhs_splits[3]))}
    return logs


def sum_nvprof_ops(nvprof_dict):
    sum_min = 0
    sum_max = 0
    sum_avg = 0
    for key, val in nvprof_dict.items():
        sum_min += val['min_ops']
        sum_max += val['max_ops']
        sum_avg += val['avg_ops']
    return sum_min, sum_max, sum_avg


def cluster_nvprof_ops(nvprof_dict, verbose=False):
    translation = {'convolve_sgemm': 'conv', 'volta_gcgemm': 'volta_gcgemm', 'relu': 'relu',
                   'fft2d_r2c': 'fft2d_r2c', 'fft2d_c2r': 'fft2d_c2r',
                   'EigenMetaKernel<Eigen::TensorEvaluator': 'EigenMetaKernel<Eigen::TensorEvaluator',
                   'stridedB': 'volta_scudnn_stridedB_splitK', 'gcgemm': 'volta_gcgemm_nt',
                   'cgemm': 'volta_cgemm_tn', 'cudnn::detail::wgrad_alg0_engine': 'cudnn::detail::wgrad_alg0_engine',
                   'volta_sgemm': 'volta_sgemm', 'void cudnn::detail::dgrad_engine': 'void cudnn::detail::dgrad_engine',
                   'void DSE::vector_fft': 'void DSE::vector_fft',
                   'void pooling_bw_kernel_max_nchw': 'void pooling_bw_kernel_max_nchw',
                   'pooling_bw_kernel': 'pooling_bw_kernel', 'pooling_fw': 'pooling_fw',
                   'winograd_nonfused': 'winograd_nonfused', 'regular_fft': 'regular_fft', 'bn_bw': 'batch_norm_bw',
                   'bn_fw': 'batch_norm_forward', }
    new_dict = dict()
    count = 0
    for key, val in nvprof_dict.items():
        grouped = False
        new_val = val.copy()
        for spec_name, gen_name in translation.items():
            if spec_name in key and not grouped:
                if verbose:
                    print(key[:30] + ' >> contains >> ' + spec_name)
                old_val = new_dict.get(gen_name, None)
                if old_val is not None:
                    if verbose:
                        print('existing entry:', old_val)
                        print('current entry:', new_val)
                    for prop_name in ['min_ops', 'max_ops', 'avg_ops', 'invocations']:
                        new_val[prop_name] += old_val[prop_name]
                else:
                    if verbose:
                        print('no prior entry found. Using current entry:', new_val)
                new_dict[gen_name] = new_val
                grouped = True
                count += 1
        if not grouped:
            if verbose:
                print('Could not group key:', key)
            new_dict[key] = new_val
        if verbose:
            print('')
    print('Collapsed {} of {} entries'.format(count, len(nvprof_dict)))
    return new_dict


def sort_nvprof_dict(nvprof_dict, sort_key='avg_ops'):
    new_dict = dict()
    for key, val_dict in nvprof_dict.items():
        new_dict[key] = val_dict[sort_key]
    sorted_dict = sorted(new_dict.items(), key=lambda x: x[1])[::-1]
    return OrderedDict(sorted_dict)


# http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf

def conv(inputs, params, bytesize):
    # NCHW not NHWC
    num_weights = np.prod(params['kernel'] + [params['features'], inputs[1]])
    outputs = tuple([inputs[0], params['features'], inputs[2] // params['stride'][0], inputs[3] // params['stride'][1]])
    mem = np.prod(outputs) * bytesize
    # account for the stride as well!
    this_ops = np.prod(list(params['kernel']) + [inputs[1]] + list(outputs[2:]) + [params['features']])
    return outputs, num_weights, mem, this_ops


def linear(inputs, params, bytesize):
    if len(inputs) == 4:
        inputs = (inputs[0], np.prod(inputs[1:]))
    outputs = (inputs[0], params['bias'])
    num_weights = params['bias'] + np.prod([params['bias'], inputs[1]]) + batch_norm(inputs)
    mem = np.prod(outputs) * bytesize
    this_ops = inputs[1] * params['weights']
    return outputs, num_weights, mem, this_ops


def pool(inputs, params, bytesize):
    outputs = (inputs[0], inputs[1], inputs[2] // params['stride'][0], inputs[3] // params['stride'][1])
    mem = np.prod(outputs) * bytesize
    return outputs, 0, mem, 0


def residual(orig_inputs, params, bytesize):
    weights = 0
    mem = 0
    ops = 0
    inputs = orig_inputs
    for layer_name, layer_params in list(params.items()):
        if not layer_name.startswith('conv'):
            continue
        outputs, curr_weights, curr_mem, curr_ops = conv(inputs, layer_params, bytesize)
        weights += curr_weights + batch_norm(outputs)
        mem += curr_mem
        ops += curr_ops
        # print('\t%s - %s, weights: %3.1e, memory: %3.1f MB, ops: %3.1e' % \
        # (layer_name, outputs, curr_weights, curr_mem / 1024**2, curr_ops))
        inputs = outputs
    # last conv for
    if outputs != orig_inputs:
        shortcut_parms = {"kernel": [1, 1], "features": outputs[1], "batch_norm": True, "stride": [1, 1]}
        orig_inputs, curr_weights, curr_mem, curr_ops = conv(orig_inputs, shortcut_parms, bytesize)
        weights += curr_weights
        mem += curr_mem
        ops += curr_ops
    return outputs, weights, mem, ops


def batch_norm(inputs):
    return 2 * inputs[1]


def calculate_network_complexity(inputs, network, is_fp16=False, verbose=False):
    bytesize = 4
    if is_fp16:
        bytesize = 2

    layer_stats = []
    tot_weights = 0
    tot_mem = np.prod(inputs) * bytesize
    tot_ops = 0

    if verbose:
        print('Inputs: %s, memory: %3.1f MB' % (inputs, tot_mem / 1024 ** 2))
    for layer_name, layer_params in list(network.items()):
        # print('-------------------------')
        # print(layer_name, layer_params)
        if layer_params['type'] == 'convolutional':
            func = conv
        elif layer_params['type'] == 'pooling':
            func = pool
        elif layer_params['type'] in ['fully_connected', 'linear_output']:
            func = linear
        elif layer_params['type'] == 'residual':
            func = residual
        else:
            print('Unrecognized layer type ' + layer_params['type'])
            break
        outputs, weights, mem, this_ops = func(inputs, layer_params, bytesize)
        if verbose:
            print('%s - %s, weights: %d, memory: %3.1f MB, ops: %3.1e' % (
                  layer_name, outputs, weights, mem / 1024 ** 2, this_ops))
        inputs = outputs
        tot_ops += this_ops
        tot_mem += mem
        tot_weights += weights
        layer_stats.append({'name': layer_name, 'shape': outputs, 'weights': weights, 'memory': mem, 'ops': this_ops,
                            'type': layer_params['type']})
    if verbose:
        print('Total # of layers: %d,  weights: %3.1e, memory: %s MB, ops: %3.2e \n' % (len(network), tot_weights,
                                                                                        tot_mem / 1024 ** 2, tot_ops))
    return layer_stats, tot_weights, tot_mem, tot_ops


import os
import re
from collections import OrderedDict
import numpy as np
import sys

FWD_ALGO_list=[
"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
"CUDNN_CONVOLUTION_FWD_ALGO_FFT",
"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"]

BWD_ALGO_DATA_list= [
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"]


BWD_ALGO_FILTER_list=["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
"CUDNN_CONVOLUTION_BWD_FILTER_WINOGRAD_NONFUSED",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING"]

FWD_ALGO_TENSORCORE=["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
]

BWD_ALGO_DATA_TENSORCORE=["CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"]

BWD_ALGO_FILTER_TENSORCORE=["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"]

MATH_OPS_list= ['CUDNN_TENSOR_OP_MATH', 'CUDNN_DEFAULT_MATH']

def todict(LIST):
    return OrderedDict([(itm, [re.compile(itm), 0]) for itm in LIST])


def count_occurences(filepath, line_bounds, ord_dict_list, portion=0.5):
    line_lb, line_ub = line_bounds
    with open(filepath,'r') as f:
        for (num_line,line) in enumerate(f):
            if num_line > line_lb and num_line < line_ub:
                for ord_dict in ord_dict_list:
                    for key, itm in ord_dict.items():
                        if itm[0].search(line):
                            ord_dict[key][1] += 1


def rank_entries(ord_dict_list, steps):
    FWD_ALGO_TENSORCORE=["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
    ]

    BWD_ALGO_DATA_TENSORCORE=["CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"]

    BWD_ALGO_FILTER_TENSORCORE=["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"]
    entries = dict()
    for ord_dict in ord_dict_list:
        arr_counts = np.array([itm[1] for _, itm in ord_dict.items()])
        indices = np.argsort(arr_counts)[::-1]
        keys = list(ord_dict.keys())
        print('Trace from training step=%d to step=%d' %(steps[0], steps[1]))
        print('CUDA FUNCTION, # CUDA CALLS, TENSORCORES USAGE')
        for ind in indices:
            algo_name = keys[ind]
            if algo_name in FWD_ALGO_TENSORCORE+BWD_ALGO_DATA_TENSORCORE+BWD_ALGO_FILTER_TENSORCORE:
                tensorcore_usage = "YES"
            else:
                tensorcore_usage = "NO"
            print('%s, %d ,%s ' %(algo_name, ord_dict[algo_name][1], tensorcore_usage))
            entries[algo_name]={'stats':ord_dict[algo_name][1], 'tensor_core':tensorcore_usage}
        print('\n')
    return entries

def get_step_timing(logfile):
    step_1 = re.compile('step= 90')
    step_2 = re.compile('step= 100')
    times, steps = [], []
    with open(logfile, mode='r') as f:
        for line in f:
            if step_1.search(line) or step_2.search(line):
                stream = line.split(',')
                time = stream[0].split('=')[-1]
                step = stream[1].split('=')[-1]
                times.append(float(time))
                steps.append(int(step))
    return times, steps

def get_lines_bounds(times, logfile):
    pattern = re.compile('Time:')
    lines = []
    with open(logfile, mode='r') as f:
        for i,line in enumerate(f):
            if pattern.search(line):
                stream=line
                stream=line.split(' ')[-3]
                time_list = re.findall('\d+',stream)
                total_time = int(time_list[0])*3600*24 + int(time_list[1])*3600 + int(time_list[2])*60 + int(time_list[3])
                if total_time > times[0] or total_time < total_time: lines.append(i)
    return lines[0], lines[-1]


def parse_cudnn_log(cuddn_logfile, train_logfile):
    # Dictionaries
    FWD_ALGO = todict(FWD_ALGO_list)
    BWD_DATA_ALGO = todict(BWD_ALGO_DATA_list)
    BWD_FILTER_ALGO = todict(BWD_ALGO_FILTER_list)
    MATH_OPS = todict(MATH_OPS_list)
    ord_dict_list = [FWD_ALGO, BWD_DATA_ALGO, BWD_FILTER_ALGO, MATH_OPS]
    # parsing
    times, steps = get_step_timing(train_logfile)
    line_lb, line_ub = get_lines_bounds(times, cudnn_logfile)
    count_occurences(cudnn_logfile, [line_lb, line_ub], ord_dict_list, portion=0.75)
    _ = rank_entries(ord_dict_list, steps)
    #TODO: save dict as CSV or don't use dict.
