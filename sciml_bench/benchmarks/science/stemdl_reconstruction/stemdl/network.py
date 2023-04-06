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

from collections import OrderedDict, deque
import re
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
import horovod.tensorflow as hvd
from .mp_wrapper import mp_regularizer_wrapper

worker_name='horovod'
tf.logging.set_verbosity(tf.logging.ERROR)

class ConvNet(object):
    """
    Vanilla Convolutional Neural Network (Feed-Forward).
    """
    def __init__(self, scope, params, hyper_params, network, images, labels, operation='train',
                 summary=False, verbose=True):
        """
        :param params: dict
        :param global_step: as it says
        :param hyper_params: dictionary, hyper-parameters
        :param network: collections.OrderedDict, specifies ConvNet layers
        :param images: batch of images
        :param labels: batch of labels
        :param operation: string, 'train' or 'eval'
        :param summary: bool, flag to write tensorboard summaries
        :param verbose: bool, flag to print shapes of outputs
        :return:
        """
        self.scope = scope
        self.params = params
        self.global_step = 0
        self.hyper_params = hyper_params
        self.network = network
        self.images = images
        if self.params['IMAGE_FP16']: #and self.images.dtype is not tf.float16 and operation == 'train':
            self.images = tf.cast(self.images, tf.float16)
        image_shape = images.get_shape().as_list()
        if self.params['TENSOR_FORMAT'] != 'NCHW' :
            # change from NHWC to NCHW format
            # TODO: add flag to swith between 2 ....
            self.images = tf.transpose(self.images, perm=[0, 3, 1, 2])
        # self.images = self.get_glimpses(self.images)
        self.labels = labels
        self.net_type = self.hyper_params['network_type']
        self.operation = operation
        self.summary = summary
        self.verbose = verbose
        self.num_weights = 0
        self.misc_ops = []
        # self.reuse = tf.AUTO_REUSE
        self.reuse = None
        if self.operation == 'eval_run':
            self.reuse = True
            self.operation == 'eval'
        elif self.operation == 'eval_ckpt':
            self.operation == 'eval'
        self.bytesize = 2
        if not self.params['IMAGE_FP16']: self.bytesize = 4
        self.mem = np.prod(self.images.get_shape().as_list()) * self.bytesize/1024  # (in KB)
        self.ops = 0
        if "batch_norm" in self.hyper_params:
            self.hyper_params["batch_norm"]["decay"] = self.hyper_params["batch_norm"].get("decay", 0.995)
            self.hyper_params["batch_norm"]["epsilon"] = self.hyper_params["batch_norm"].get("epsilon", 1E-5)
        else:
            # default params
            self.hyper_params["batch_norm"] = {"epsilon": 1E-5, "decay": 0.995}
        self.model_output = None
        self.scopes = []

        # self.initializer = self._get_initializer(hyper_params.get('initializer', None))

    def print_rank(self, *args, **kwargs):
        if hvd.rank() == 0 and self.operation == 'train':
            print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            self.print_rank(*args, **kwargs)

    def build_model(self, summaries=False):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        self.initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        # Initiate 1st layer
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        layer_name, layer_params = list(self.network.items())[0]
        with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
            out, kernel = self._conv(input=self.images, params=layer_params)
            do_bn = layer_params.get('batch_norm', False)
            if do_bn:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=layer_params)
            out = self._activate(input=out, name=scope.name, params=layer_params)
            in_shape = self.images.get_shape().as_list()
            # Tensorboard Summaries
            if self.summary:
                self._activation_summary(out)
                self._activation_image_summary(out)
                self._kernel_image_summary(kernel)

            self._print_layer_specs(layer_params, scope, in_shape, out.get_shape().as_list())
            self.scopes.append(scope)

        # Initiate the remaining layers
        for layer_name, layer_params in list(self.network.items())[1:]:
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
                if layer_params['type'] == 'convolutional':
                    out, _ = self._conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    out = self._pool(input=out, name=scope.name, params=layer_params)

                if layer_params['type'] not in ['convolutional', 'pooling', 'fully_connected', 'linear_output']:
                    out = self._compound_layer(out, layer_params, scope)
                    # Continue any summary
                    if self.summary: self._activation_summary(out)

                # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + ' was not implemented!')
                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
            self.scopes.append(scope)
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(self.network),
                                                                                        self.num_weights,
                                                                                        format(self.mem / 1024),
                                                                                        self.get_ops()))
        self.model_output = out

    def _compound_layer(self, out, layer_params, scope):
        """
        Handles the computation of more complex layer types such as Residual blocks, Inception, etc.

        :param out: 4D tensor, Input to the layer
        :param layer_params: OrderedDictionary, Parameters for the layer
        :param scope: str, name of the layer
        :return: 4D tensor, output of the layer
        """
        pass

    # @staticmethod
    def _get_initializer(self, params):
        """
        Returns an Initializer object for initializing weights

        Note - good reference:
         - https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/keras/_impl/keras/initializers.py
        :param params: dictionary
        :return: initializer object
        """
        if params is not None:
            if isinstance(params, dict):
                params_copy = params.copy()
                name = str(params_copy.pop('type').lower())
                if name == 'uniform_unit_scaling':
                    # self.print_verbose('using ' + name + ' initializer')
                    # Random walk initialization (currently in the code).
                    return tf.initializers.variance_scaling(distribution="uniform")
                elif name == 'truncated_normal':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.truncated_normal_initializer(**params_copy)
                elif name == 'variance_scaling':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.initializers.variance_scaling(distribution="normal")
                elif name == 'random_normal':
                    # self.print_verbose('using ' + name + ' initializer')
                    # Normalized Initialization ( eq. 16 in Glorot et al.).
                    return tf.random_normal_initializer(**params_copy)
                elif name == 'random_uniform':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.random_uniform_initializer(**params_copy)
                elif name == 'xavier':  # Glorot uniform initializer, also called Xavier
                    # http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.contrib.layers.xavier_initializer(**params_copy)
                elif name in ['he', 'lecun']:
                    """
                    Note that tf.variance_scaling_initializer and tf.contrib.layers.variance_scaling_initializer
                    take the same kinds of parameters with different names and formats.

                    However, tf.variance_scaling_initializer doesn't seem to be available on TF 1.2.1 on the DGX1
                    """
                    params_copy['factor'] = params_copy.pop('scale', 1)
                    params_copy['uniform'] = params_copy.pop('distribution', True)
                    if 'uniform' in params_copy:
                        if isinstance(params_copy['uniform'], str):
                            if params_copy['uniform'].lower() == 'uniform':
                                params_copy['uniform'] = True
                            else:
                                params_copy['uniform'] = False
                    if name == 'he':
                        # He et al., http://arxiv.org/abs/1502.01852
                        _ = params_copy.pop('factor', None)  # force it to be 2.0 (default anyway)
                        _ = params_copy.pop('mode', None)  # force it to be 'FAN_IN' (default anyway)
                        # uniform parameter is False by default -> normal distribution
                        # self.print_verbose('using ' + name + ' initializer')
                        return tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', **params_copy)
                    elif name == 'lecun':
                        _ = params_copy.pop('factor', None)  # force it to be 1.0
                        _ = params_copy.pop('mode', None)  # force it to be 'FAN_IN' (default anyway)
                        # uniform parameter is False by default -> normal distribution
                        # self.print_verbose('using ' + name + ' initializer')
                        return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', **params_copy)
                    self.print_verbose('Requested initializer: ' + name + ' has not yet been implemented.')
        # default = Xavier:
        # self.print_verbose('Using default Xavier instead')
        return tf.contrib.layers.xavier_initializer()


    def get_loss(self):
        # with tf.variable_scope(self.scope, reuse=self.reuse) as scope:
        if self.net_type == 'hybrid': self._calculate_loss_hybrid()
        if self.net_type == 'regressor': self._calculate_loss_regressor()
        if self.net_type == 'classifier' : self._calculate_loss_classifier()
        if self.hyper_params['langevin'] : self.add_stochastic_layer()

    def get_output(self):
        layer_params={'bias':self.labels.get_shape().as_list()[-1], 'weights':self.labels.get_shape().as_list()[-1],
            'regularize':True}
        with tf.variable_scope('linear_output', reuse=self.reuse) as scope:
            output = self._linear(input=self.model_output, name=scope.name, params=layer_params)
            print(output.name)
        if self.params['IMAGE_FP16']:
            output = tf.cast(output, tf.float32)
            return output
        return output

    #TODO: return ops per type of layer
    def get_ops(self):
        return 3*self.ops # 3 is for derivate w/t kernel + derivative w/t data + conv (*ignoring everything else eventhough they're calculated)

    def get_misc_ops(self):
        ops = tf.group(*self.misc_ops)
        return ops

    # Loss calculation and regularization helper methods

    def _calculate_loss_hybrid(self):
        dim = self.labels.get_shape().as_list()[-1]
        num_classes = self.params['NUM_CLASSES']
        if self.hyper_params['langevin']:
            class_labels = self.labels
            if class_labels.dtype is not tf.int64:
                class_labels = tf.cast(class_labels, tf.int64)
            regress_labels = tf.random_normal(class_labels.get_shape().as_list(), stddev=0.01, dtype=tf.float64)
        else:
            regress_labels, class_labels = tf.split(self.labels,[dim-num_classes, num_classes],1)
        outputs = []
        for layer_name, labels in zip(['linear_output', 'stochastic'],
                                            [class_labels, regress_labels]):
            layer_params={'bias':labels.get_shape().as_list()[-1], 'weights':labels.get_shape().as_list()[-1],
                'regularize':True}
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                out = tf.cast(self._linear(input=self.model_output, name=scope.name, params=layer_params), tf.float32)
                print(out.name)
            self.print_rank('Output Layer : %s' %format(out.get_shape().as_list()))
            outputs.append(out)
        mixing = self.hyper_params['mixing']
        cost = (1-mixing)*self._calculate_loss_classifier(net_output=outputs[0], labels=class_labels) + \
                        mixing*self._calculate_loss_regressor(net_output=outputs[1],
                        labels=regress_labels, weight=mixing)
        return cost

    def _calculate_loss_regressor(self, net_output=None, labels=None, weight=None):
        """
        Calculate the loss objective for regression
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        if net_output is None:
            net_output = self.get_output()
        if labels is None:
            labels = self.labels
        if weight is None:
            weight = 1.0
        params = self.hyper_params['loss_function']
        assert params['type'] == 'Huber' or params['type'] == 'MSE' \
        or params['type'] == 'LOG', "Type of regression loss function must be 'Huber' or 'MSE'"
        if params['type'] == 'Huber':
            # decay the residual cutoff exponentially
            decay_steps = int(self.params['NUM_EXAMPLES_PER_EPOCH'] / self.params['batch_size'] \
                              * params['residual_num_epochs_decay'])
            initial_residual = params['residual_initial']
            min_residual = params['residual_minimum']
            decay_residual = params['residual_decay_factor']
            residual_tol = tf.train.exponential_decay(initial_residual, self.global_step, decay_steps,
                                                      decay_residual, staircase=False)
            # cap the residual cutoff to some min value.
            residual_tol = tf.maximum(residual_tol, tf.constant(min_residual))
            if self.summary:
                tf.summary.scalar('residual_cutoff', residual_tol)
            # calculate the cost
            cost = tf.losses.huber_loss(labels, weights=weight, predictions=net_output, delta=residual_tol,
                                        reduction=tf.losses.Reduction.MEAN)
        if params['type'] == 'MSE':
            cost = tf.losses.mean_squared_error(labels, weights=weight, predictions=net_output,
                                                reduction=tf.losses.Reduction.MEAN)
        if params['type'] == 'LOG':
            cost = tf.losses.log_loss(labels, weights=weight, predictions=net_output, reduction=tf.losses.Reduction.MEAN)
        return cost

    def _calculate_loss_classifier(self, net_output=None, labels=None, weight=None):
        """
        Calculate the loss objective for classification
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        if labels is None:
            labels = self.labels
        if labels.dtype is not tf.int64:
            labels = tf.cast(labels, tf.int64)
        if net_output is None:
            net_output = self.get_output()
        if weight is None:
            weight = 1.0
        labels = tf.argmax(labels, axis=1)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=net_output)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        precision_1 = tf.scalar_mul(1. / self.params['batch_size'],
                                    tf.reduce_sum(tf.cast(tf.nn.in_top_k(net_output, labels, 1), tf.float32)))
        precision_5 = tf.scalar_mul(1. / self.params['batch_size'],
                                    tf.reduce_sum(tf.cast(tf.nn.in_top_k(net_output, labels, 5), tf.float32)))
        if self.summary :
            tf.summary.scalar('precision@1_train', precision_1)
            tf.summary.scalar('precision@5_train', precision_5)
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
        return cross_entropy_mean

    # Network layers helper methods
    def _conv(self, input=None, params=None):
        """
        Builds 2-D convolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of convolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features']
        kernel_shape = list(params['kernel']) + [input.shape[1].value, features]

        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))

            self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only

        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        output = tf.nn.conv2d(input, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024
        # batch * width * height * in_channels * kern_h * kern_w * features
        # input = batch_size (ignore), channels, height, width
        # http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf
        # this_ops = np.prod(params['kernel'] + input.get_shape().as_list()[1:] + [features])
        # self.print_rank('\tops: %3.2e' % (this_ops))
        """
        # batch * width * height * in_channels * (kern_h * kern_w * channels)
        # at each location in the image:
        ops_per_conv = 2 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of convolutions on the image for a single filter / output channel (stride brings down the number)
        convs_per_filt = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = filters * convs/filter * ops/conv
        this_ops = np.prod([params['features'], convs_per_filt, ops_per_conv])
        if verbose:
            self.print_verbose('\t%d ops/conv, %d convs/filter, %d filters = %3.2e ops' % (ops_per_conv, convs_per_filt,
                                                                              params['features'], this_ops))
        """
        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input.get_shape().as_list()[1:])
        self.ops += this_ops

        return output, kernel

    def _deconv(self, input=None, params=None, verbose=True):
        """
        Builds 2-D deconvolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of deconvolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features']
        kernel_shape = list(params['kernel']) + [features, input.shape[1].value]

        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            if verbose:
                self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only
        upsample = params['upsample']
        output_shape = [input.shape[0].value, features, input.shape[2].value*upsample, input.shape[2].value*upsample]
        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        output = tf.nn.conv2d_transpose(input, kernel, output_shape, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights, memory, and flops
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024
        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input.get_shape().as_list()[1:])
        self.ops += this_ops

        return output, kernel

    def _add_bias(self, input=None, params=None):
        """
        Adds bias to a convolutional layer.
        :param input:
        :param params:
        :return:
        """
        bias_shape = input.shape[-1].value
        bias = self._cpu_variable_init('bias', shape=bias_shape, initializer=tf.zeros_initializer())
        output = tf.nn.bias_add(input, bias)

        # Keep tabs on the number of bias parameters and memory
        self.num_weights += bias_shape
        self.mem += bias_shape*self.bytesize / 1024
        # self.ops += bias_shape
        return output

    def _batch_norm(self, input=None, reuse=None):
        """
        Batch normalization
        :param input: as it says
        :return:
        """
        # Initializing hyper_parameters
        shape = [input.shape[1].value]
        epsilon = self.hyper_params["batch_norm"]["epsilon"]
        decay = self.hyper_params["batch_norm"]["decay"]
        is_training = 'train' == self.operation
        # TODO: scaling and centering during normalization need to be hyperparams. Now hardwired.
        param_initializers={
              'beta': tf.constant_initializer(0.0),
              'gamma': tf.constant_initializer(0.1),
        }
        output = tf.contrib.layers.batch_norm(input, decay=decay, scale=True, epsilon=epsilon,zero_debias_moving_mean=False,is_training=is_training,fused=True,data_format='NCHW',renorm=False,param_initializers=param_initializers)
        # output = input
        # Keep tabs on the number of weights
        self.num_weights += 2 * shape[0]  # scale and offset (beta, gamma)
        # consistently ignored by most papers / websites for ops calculation
        return output

    def _linear(self, input=None, params=None, name=None, verbose=True):
        """
        Linear layer
        :param input:
        :param params:
        :return:
        """
        assert params['weights'] == params['bias'], " weights and bias outer dimensions do not match"
        input_reshape = tf.reshape(input,[self.params['batch_size'], -1])
        dim_input = input_reshape.shape[1].value
        weights_shape = [dim_input, params['weights']]
        init_val = max(np.sqrt(2.0/params['weights']), 0.01)
        self.print_verbose('stddev: %s' % format(init_val))
        bias_shape = [params['bias']]

        # Fine tuning the initializer:
        lin_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(lin_initializer, tf.uniform_unit_scaling_initializer):
            if params['type'] == 'fully_connected':
                if params['activation'] == 'tanh':
                    lin_initializer.factor = 1.15
                elif params['activation'] == 'relu':
                    lin_initializer.factor = 1.43
            elif params['type'] == 'linear_output':
                lin_initializer.factor = 1.0
        elif isinstance(lin_initializer, tf.random_normal_initializer):
            init_val = max(np.sqrt(2.0 / params['weights']), 0.01)
            if verbose:
                self.print_verbose('stddev: %s' % format(init_val))
            lin_initializer.mean = 0.0
            lin_initializer.stddev = init_val

        weights = self._cpu_variable_init('weights', shape=weights_shape, initializer=tf.random_normal_initializer(0,0.01),
                                          regularize=params['regularize'])
        bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.constant_initializer(1.e-3))
        output = tf.nn.bias_add(tf.matmul(input_reshape, weights), bias, name=name)

        # Keep tabs on the number of weights and memory
        self.num_weights += bias_shape[0] + np.cumprod(weights_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1] * self.bytesize / 1024
        # equation =  inputs * weights + bias for a single example
        # equation = [features] * [features, nodes]
        # http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf
        # this_ops = input.get_shape().as_list()[1] * params['weights']
        # self.print_verbose('\tops: %3.2e' % (this_ops))
        """
        # for each element in the output - feature^2 multiplies + feature^2 additions
        ops_per_element = 2 * dim_input ** 2
        # number of elements in outputs = batch * hidden nodes in this layer
        num_dot_prods = params['weights']  # * self.params['batch_size
        # addition of bias = nodes number of additions
        bias_ops = params['weights']
        # batch * nodes * 2 * features + nodes <- 2 comes in for the dot prod + sum
        this_ops = ops_per_element * num_dot_prods + bias_ops
        if verbose:
            self.print_verbose('\t%d ops/element, %d dot products, %d additions for bias = %3.2e ops' % (ops_per_element,
                                                                                                  num_dot_prods,
                                                                                            bias_ops, this_ops))
        """
        # self.ops += this_ops
        return output

    def _dropout(self, input=None, keep_prob=0.5, params=None, name=None):
        """
        Performs dropout
        :param input:
        :param params:
        :param name:
        :return:
        """
        return tf.nn.dropout(input, keep_prob=tf.constant(keep_prob, dtype=input.dtype))

    def _activate(self, input=None, params=None, name=None, verbose=False):
        """
        Activation
        :param input: as it says
        :param params: dict, must specify activation type
        :param name: scope.name
        :return:
        """
        # should ignore the batch size in the calculation!
        # this_ops = 2 * np.prod(input.get_shape().as_list()[1:])
        if verbose:
            self.print_verbose('\tactivation = %3.2e ops' % this_ops)
        # self.ops += this_ops

        if params is not None:
            if params['activation'] == 'tanh':
                return tf.nn.tanh(input, name=name)

        return tf.nn.relu(input, name=name)

    def _pool(self, input=None, params=None, name=None, verbose=True):
        """
        Pooling
        :param params: dict, must specify type of pooling (max, average), stride, and kernel size
        :return:
        """
        stride_shape = [1,1]+params['stride']
        kernel_shape = [1,1]+params['kernel']
        if params['pool_type'] == 'max':
            output = tf.nn.max_pool(input, kernel_shape, stride_shape, params['padding'], name=name, data_format='NCHW')
        if params['pool_type'] == 'avg':
            output = tf.nn.avg_pool(input, kernel_shape, stride_shape, params['padding'], name=name, data_format='NCHW')

        # Keep tabs on memory
        self.mem += np.cumprod(output.get_shape().as_list())[-1] * self.bytesize / 1024

        # at each location in the image:
        # avg: 1 to sum each of the N element, 1 op for avg
        # max: N max() operations
        ops_per_pool = 1 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of pools on the image for a single filter / output channel (stride brings down the number)
        num_pools = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = num images * filters * convs/filter * ops/conv
        # self.print_verbose('\t%d ops/pool, %d pools = %3.2e ops' % (ops_per_pool, num_pools,
        #                                                    num_pools * ops_per_pool))

        self.ops += num_pools * ops_per_pool

        return output

    # Summary helper methods
    @staticmethod
    def _activation_summary(x):
        """Helper to create summaries for activations.

         Creates a summary that provides a histogram of activations.
         Creates a summary that measures the sparsity of activations.

         Args:
           x: Tensor
         Returns:
           nothing
        """
        # Remove 'worker_[0-9]/' from the name in Tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % worker_name, '', x.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    @staticmethod
    def _activation_image_summary(image_stack, n_features=None):
        """ Helper to show images of activation maps in summary.

        Args:
            image_stack: Tensor, 4-d output of conv/pool layer
            n_features: int, # of featuers to display, Optional, default is half of features depth.
        Returns:
            Nothing
        """

        # Transpose to NHWC
        image_stack = tf.transpose(image_stack, perm=[0, 2, 3, 1])
        #
        tensor_name = re.sub('%s_[0-9]*/' % worker_name, '', image_stack.name)
        # taking only first 3 images from batch
        if n_features is None:
            # nFeatures = int(pool.shape[-1].value /2)
            n_features = -1
        for ind in range(1):
            map = tf.slice(image_stack, (ind, 0, 0, 0), (1, -1, -1, n_features))
            map = tf.reshape(map, (map.shape[1].value, map.shape[2].value, map.shape[-1].value))
            map = tf.transpose(map, (2, 0 , 1))
            map = tf.reshape(map, (-1, map.shape[1].value, map.shape[2].value, 1))

            # Tiling
            nOfSlices = map.shape[0].value
            n = int(np.ceil(np.sqrt(nOfSlices)))
            # padding by 4 pixels
            padding = [[0, n ** 2 - nOfSlices], [0, 4], [0, 4], [0, 0]]
            map_padded = tf.pad(map, paddings=padding)
            # reshaping and transposing gymnastics ...
            new_shape = (n, n) + (map_padded.shape[1].value, map_padded.shape[2].value, 1)
            map_padded = tf.reshape(map_padded, new_shape)
            map_padded = tf.transpose(map_padded, perm=(0, 2, 1, 3, 4))
            new_shape = (n * map_padded.shape[1].value, n * map_padded.shape[3].value, 1)
            map_tile = tf.reshape(map_padded, new_shape)
            # Convert to 4-d
            map_tile = tf.expand_dims(map_tile,0)
            map_tile = tf.log1p(map_tile)
            # Display feature maps
            tf.summary.image(tensor_name + '/activation'+ str(ind), map_tile)

    @staticmethod
    def _kernel_image_summary(image_stack, n_features=None):
        """ Helper to show images of activation maps in summary.

        Args:
            image_stack: Tensor, 4-d output of conv/pool layer
            n_features: int, # of featuers to display, Optional, default is half of features depth.
        Returns:
            Nothing
        """
        # Remove 'worker_[0-9]/' from the name in Tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % worker_name, '', image_stack.name)
        if n_features is None:
            n_features = -1
        map = tf.slice(image_stack, (0, 0, 0, 0), (-1, -1, -1, n_features))
        # self.print_rank('activation map shape: %s' %(format(map.shape)))
        map = tf.reshape(map, (map.shape[0].value, map.shape[1].value, map.shape[-2].value*map.shape[-1].value))
        map = tf.transpose(map, (2, 0, 1))
        map = tf.reshape(map, (-1, map.shape[1].value, map.shape[2].value, 1))
        # color_maps = tf.image.grayscale_to_rgb(map)
        # Tiling
        nOfSlices = map.shape[0].value
        n = int(np.ceil(np.sqrt(nOfSlices)))
        # padding by 4 pixels
        padding = [[0, n ** 2 - nOfSlices], [0, 4], [0, 4], [0, 0]]
        map_padded = tf.pad(map, paddings=padding)
        # reshaping and transposing gymnastics ...
        new_shape = (n, n) + (map_padded.shape[1].value, map_padded.shape[2].value, 1)
        map_padded = tf.reshape(map_padded, new_shape)
        map_padded = tf.transpose(map_padded, perm=(0, 2, 1, 3, 4))
        new_shape = (n * map_padded.shape[1].value, n * map_padded.shape[3].value, 1)
        map_tile = tf.reshape(map_padded, new_shape)
        # Convert to 4-d
        map_tile = tf.expand_dims(map_tile, 0)
        map_tile = tf.log1p(map_tile)
        # Display feature maps
        tf.summary.image(tensor_name + '/kernels' , map_tile)

    def _print_layer_specs(self, params, scope, input_shape, output_shape):
        mem_in_MB = np.cumprod(output_shape)[-1] * self.bytesize / 1024**2
        if params['type'] == 'convolutional':
            self.print_verbose('%s --- output: %s, kernel: %s, stride: %s, # of weights: %s,  memory: %s MB' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride']), format(self.num_weights), format(mem_in_MB)))
        if params['type'] == 'pooling':
            self.print_verbose('%s --- output: %s, kernel: %s, stride: %s, memory: %s MB' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride']), format(mem_in_MB)))
        if params['type'] == 'fully_connected' or params['type'] == 'linear_output':
            self.print_verbose('%s --- output: %s, weights: %s, bias: %s, # of weights: %s,  memory: %s MB' %
                   (scope.name, format(output_shape), format(params['weights']),
                     format(params['bias']), format(self.num_weights), format(mem_in_MB)))

    def _add_loss_summaries(self, total_loss, losses):
        """
        Add summaries for losses in model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        :param total_loss:
        :param losses:
        :return: loss_averages_op
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss;
        if self.summary:
            for l in losses + [total_loss]:
                # Name each loss as '(raw)' and name the moving average version of the loss
                # as the original loss name.
                loss_name = re.sub('%s_[0-9]*/' % worker_name, '', l.op.name)
                tf.summary.scalar(loss_name + ' (raw)', l)
                tf.summary.scalar(loss_name, loss_averages.average(l))

        return loss_averages_op

    def _json_summary(self):
        """
        Generate text summary out of *.json file input
        :return: None
        """
        net_list = [[key, str([self.network[key]])] for key in self.network.iterkeys()]
        hyp_list = [[key, str([self.hyper_params[key]])] for key in self.hyper_params.iterkeys()]
        net_config = tf.constant(net_list, name='network_config')
        hyp_params = tf.constant(hyp_list, name='hyper_params')
        tf.summary.text(net_config.op.name, net_config)
        tf.summary.text(hyp_params.op.name, hyp_params)
        return None

    # Variable placement, initialization, regularization
    def _cpu_variable_init(self, name, shape, initializer, trainable=True, regularize=True):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        # if self.params['IMAGE_FP16'] and self.operation == 'train':
        if self.params['IMAGE_FP16']:
            dtype = tf.float16
        else:
            dtype = tf.float32

        if regularize:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable,
                                  regularizer=self._weight_decay)
        else:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)

        return var


    def _weight_decay(self, tensor):
        return tf.multiply(tf.nn.l2_loss(tf.cast(tensor, tf.float32)), self.hyper_params['weight_decay'])

    def get_glimpses(self, batch_images):
        """
        Get bounded glimpses from images, corresponding to ~ 2x1 supercell
        :param batch_images: batch of training images
        :return: batch of glimpses
        """
        if self.params['glimpse_mode'] not in ['uniform', 'normal', 'fixed']:
            """
            print('No image glimpsing will be performed since mode: "{}" is not'
                   'among "uniform", "normal", "fixed"'
                   '.'.format(self.params['glimpse_mode']))
            """
            return batch_images

        # set size of glimpses
        #TODO: change calls to image specs from self.params to self.features
        y_size, x_size = self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH']
        crop_y_size, crop_x_size = self.params['CROP_HEIGHT'], self.params['CROP_WIDTH']
        size = tf.constant(value=[crop_y_size, crop_x_size],
                           dtype=tf.int32)

        if self.params['glimpse_mode'] == 'uniform':
            # generate uniform random window centers for the batch with overlap with input
            y_low, y_high = int(crop_y_size / 2), int(y_size - crop_y_size // 2)
            x_low, x_high = int(crop_x_size / 2), int(x_size - crop_x_size // 2)
            cen_y = tf.random_uniform([self.params['batch_size']], minval=y_low, maxval=y_high)
            cen_x = tf.random_uniform([self.params['batch_size']], minval=x_low, maxval=x_high)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        elif self.params['glimpse_mode'] == 'normal':
            # generate normal random window centers for the batch with overlap with input
            cen_y = tf.random_normal([self.params['batch_size']], mean=y_size // 2, stddev=self.params['glimpse_normal_off_stdev'])
            cen_x = tf.random_normal([self.params['batch_size']], mean=x_size // 2, stddev=self.params['glimpse_normal_off_stdev'])
            offsets = tf.stack([cen_y, cen_x], axis=1)

        elif self.params['glimpse_mode'] == 'fixed':
            # fixed crop
            cen_y = np.ones((self.params['batch_size'],), dtype=np.int32) * self.params['glimpse_height_off']
            cen_x = np.ones((self.params['batch_size'],), dtype=np.int32) * self.params['glimpse_width_off']
            offsets = np.vstack([cen_y, cen_x]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)

        else:
            # should not come here:
            return batch_images

        # extract glimpses
        glimpse_batch = tf.image.extract_glimpse(batch_images, size, offsets, centered=False, normalized=False,
                                                 uniform_noise=False, name='batch_glimpses')
        return glimpse_batch


class ResNet(ConvNet):

    # def __init__(self, *args, **kwargs):
    #     super(ResNet, self).__init__(*args, **kwargs)

    def _add_branches(self, hidden, out, verbose=True):
        """
        Adds two 4D tensors ensuring that the number of channels is consistent

        :param hidden: 4D tensor, one branch of inputs in the final step of a ResNet (more number of channels)
        :param out: 4D tensor, another branch of inputs in the final step of a ResNet (fewer number of channels)
        :param verbose: bool, (Optional) - whether or not to print statements.
        :return: 4D tensor, output of the addition
        """
        if out.get_shape().as_list()[1] != hidden.get_shape().as_list()[1]:
            # Need to do a 1x1 conv layer on the input to increase the number of channels:
            shortcut_parms = {"kernel": [1, 1], "stride": [1, 1], "padding": "SAME",
                              "features": hidden.get_shape().as_list()[1], "batch_norm": True}
            if verbose:
                self.print_verbose('Doing 1x1 conv on output to bring channels from %d to %d' % (out.get_shape().as_list()[1],
                                                                                    hidden.get_shape().as_list()[1]))
            with tf.variable_scope("shortcut", reuse=self.reuse) as scope:
                out, _ = self._conv(input=out, params=shortcut_parms)
        # ops just for the addition operation - ignore the batch size
        # this_ops = np.prod(out.get_shape().as_list()[:1])
        # if verbose:
        #     self.print_verbose('\tops for adding shortcut: %3.2e' % this_ops)
        # self.ops += this_ops
        # Now add the hidden with input
        return tf.add(out, hidden)

    def _compound_layer(self, out, layer_params, scope_name):
        if layer_params['type'] == 'residual':
            return self._residual_block(out, layer_params)

    def _residual_block(self, out, res_block_params, verbose=True):
        """
        Unit residual block consisting of arbitrary number of convolutional layers, each specified by its own
        OrderedDictionary in the parameters.
        Implementation here based on: https://arxiv.org/pdf/1603.05027.pdf
        Input >> BN >> Relu >> weight >> BN >> ReLU >> Weight >> Add Input

        :param out: 4D tensor, Input to the residual block
        :param res_block_params: OrderedDictionary, Parameters for the residual block
        :param verbose: bool, (Optional) - whether or not to print statements.
        :return: 4D tensor, output of the residual block
        """
        ops_in = self.ops

        with tf.variable_scope("pre_conv1", reuse=self.reuse) as sub_scope:
            hidden = self._batch_norm(input=out)
            hidden = self._activate(input=hidden, name=sub_scope.name)

        # First find the names of all conv layers inside
        layer_ids = []
        for parm_name in res_block_params.keys():
            if isinstance(res_block_params[parm_name], OrderedDict):
                if res_block_params[parm_name]['type'] == 'convolutional':
                    layer_ids.append(parm_name)
        """
        if verbose:
            print('internal layers:' + str(layer_ids))
            print('Working on the first N-1 layers')
        """
        # Up to N-1th layer: weight >> BN >> ReLU
        for layer_name in layer_ids[:-1]:
            if verbose:
                self.print_verbose('weight >> BN >> ReLU on layer: ' + layer_name)
            with tf.variable_scope(layer_name, reuse=self.reuse) as sub_scope:
                layer_params = res_block_params[layer_name]
                hidden, _ = self._conv(input=hidden, params=layer_params)
                hidden = self._batch_norm(input=hidden)
                hidden = self._activate(input=hidden, name=sub_scope.name, params=layer_params)

        if verbose:
            self.print_verbose('weight ONLY on layer: ' + layer_ids[-1])
        # last layer: Weight ONLY
        with tf.variable_scope(layer_ids[-1], reuse=self.reuse) as sub_scope:
            hidden, _ = self._conv(input=hidden, params=res_block_params[layer_ids[-1]])

        # Now add the two branches
        ret_val = self._add_branches(hidden, out)

        if verbose:
            ops_out = self.ops
            self.print_verbose('\tresnet ops = %3.2e' % (ops_out - ops_in))

        return ret_val

    def _print_layer_specs(self, params, scope, input_shape, output_shape):
        if params['type'] == 'pooling':
            self.print_verbose('%s --- output: %s, kernel: %s, stride: %s' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride'])))
        elif params['type'] == 'residual':
            mem_in_MB = np.cumprod(output_shape)[-1] * self.bytesize / 1024**2
            self.print_verbose('Residual Layer: ' + scope.name)
            for parm_name in params.keys():
                if isinstance(params[parm_name], OrderedDict):
                    if params[parm_name]['type'] == 'convolutional':
                        conv_parms = params[parm_name]
                        self.print_verbose('\t%s --- output: %s, kernel: %s, stride: %s, # of weights: %s,  memory: %s MB' %
                              (parm_name, format(output_shape), format(conv_parms['kernel']),
                               format(conv_parms['stride']), format(self.num_weights), format(0)))


        else:
            super(ResNet, self)._print_layer_specs(params, scope, input_shape, output_shape)


class FCDenseNet(ConvNet):
    """
    Fully Convolutional Dense Neural Network
    """

    def build_model(self, summaries=False):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        # Initiate 1st layer
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        layer_name, layer_params = list(self.network.items())[0]
        with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
            out, kernel = self._conv(input=self.images, params=layer_params)
            do_bn = layer_params.get('batch_norm', False)
            if do_bn:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=layer_params)
            out = self._activate(input=out, name=scope.name, params=layer_params)
            in_shape = self.images.get_shape().as_list()
            # Tensorboard Summaries
            if self.summary:
                self._activation_summary(out)
                self._activation_image_summary(out)
                self._kernel_image_summary(kernel)

            self._print_layer_specs(layer_params, scope, in_shape, out.get_shape().as_list())
            self.scopes.append(scope)

        # Initiate the remaining layers
        skip_connection_list = list()
        block_upsample_list = list()
        skip_hub = -1
        for layer_name, layer_params in list(self.network.items())[1:]:
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
                if layer_params['type'] == 'convolutional':
                    self.print_verbose(">>> Adding Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._conv(input=out, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    self.print_verbose(">>> Adding Pooling Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._pool(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'linear_output':
                    in_shape = out.get_shape().as_list()
                    # sometimes the same network json file is used for regression and classification.
                    # Taking the number of classes from the parameters / flags instead of the network json
                    if layer_params['bias'] != self.params['NUM_CLASSES']:
                        self.print_verbose("Overriding the size of the bias ({}) and weights ({}) with the 'NUM_CLASSES' parm ({})"
                              "".format(layer_params['bias'], layer_params['weights'], self.params['NUM_CLASSES']))
                        layer_params['bias'] = self.params['NUM_CLASSES']
                        layer_params['weights'] = self.params['NUM_CLASSES']
                    out = self._linear(input=out, name=scope.name, params=layer_params)
                    assert out.get_shape().as_list()[-1] == self.params['NUM_CLASSES'], 'Dimensions of the linear output layer' + \
                                                                         'do not match the expected output set in the params'
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'dense_block_down':
                    self.print_verbose(">>> Adding Dense Block Down: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._dense_block(out, layer_params, scope)
                    skip_connection_list.append(out)
                    skip_hub += 1
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'dense_block_bottleneck':
                    self.print_verbose(">>> Adding Dense Block Bottleneck: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, block_features = self._dense_block(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'dense_block_up':
                    self.print_verbose(">>> Adding Dense Block Up: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, block_features = self._dense_block(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'transition_up':
                    self.print_verbose(">>> Adding Transition Up: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._transition_up(block_features, skip_connection_list[skip_hub], layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    skip_hub -= 1

                if layer_params['type'] == 'transition_down':
                    self.print_verbose(">>> Adding Transition Down: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._transition_down(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + 'is not implemented!')
                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
            self.scopes.append(scope)
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(self.network),
                                                                                        self.num_weights,
                                                                                        format(self.mem / 1024),
                                                                                        self.get_ops()))
        self.model_output = tf.cast(out, tf.float32)

    def get_loss(self):
        with tf.variable_scope(self.scope, reuse=self.reuse) as scope:
            self._calculate_loss_regressor()

    def _transition_up(self, input, block_connect, layer_params, scope):
        """
        Transition up block : transposed deconvolution.
        Also add skip connection from skip hub to current output
        """
        out, _ = self._deconv(input, layer_params['deconv'], scope)
        out = tf.concat([out,block_connect], axis=1)
        return out

    def _transition_down(self, input, layer_params, scope):
        """
        TransitionDown Unit for FCDenseNet
        BN >> ReLU >> 1x1 Convolution >> Dropout >> Pooling
        """
        # BN >> RELU >> 1x1 conv >> Dropout
        conv_layer_params = layer_params['conv']
        out = self._batch_norm(input=input)
        out = self._activate(input=out, params=conv_layer_params)
        out, _ = self._conv(input=out, params=conv_layer_params)
        keep_prob = layer_params.get('dropout', None)
        if keep_prob is not None:
            out = self._dropout(input=out, name=scope.name+ '_dropout')
        # Pooling
        pool_layer_params = layer_params['pool']
        out = self._pool(input=out, params=pool_layer_params)
        in_shape = input.get_shape().as_list()
        out_shape = out.get_shape().as_list()
        self._print_layer_specs(pool_layer_params, scope, in_shape, out_shape)
        return out

    def _db_layer(self, input, layer_params, scope):
        """
        Dense Block unit for DenseNets
        BN >> Nonlinear Activation >> Convolution >> Dropout
        """
        #out = self._batch_norm(input=input)
        out = self._activate(input=input, params=layer_params)
        out, _ = self._conv(input=out, params=layer_params)
        keep_prob = layer_params.get('dropout', None)
        if keep_prob is not None:
            out = self._dropout(input=out, name=scope.name+ '_dropout')
        in_shape = input.get_shape().as_list()
        out_shape = out.get_shape().as_list()
        self._print_layer_specs(layer_params, scope, in_shape, out_shape)
        return out

    def _dense_block(self, input, layer_params, scope):
        """
        Returns output, block_features (feature maps created)
        """
        # First find the names of all conv layers inside
        layer_params = layer_params['conv']
        layer_ids = []
        for parm_name in layer_params.keys():
            if isinstance(layer_params[parm_name], OrderedDict):
                if layer_params[parm_name]['type'] == 'convolutional':
                    layer_ids.append(parm_name)
        # Build layer by layer
        block_features = []
        for layer_name in layer_ids:
            with tf.variable_scope(layer_name, reuse = self.reuse) as scope:
                # build layer
                layer = self._db_layer(input, layer_params[layer_name], scope)
                # append to list of features
                block_features.append(layer)
                #stack new layer
                input = tf.concat([input, layer], axis=1)
        block_features = tf.concat(block_features, axis=1)
        return input, block_features
