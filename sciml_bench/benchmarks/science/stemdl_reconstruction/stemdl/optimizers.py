# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optimizer ops for use in layers and tf.learn."""

# This file was copy-pasted from TF repo on 10/04/2017 by Oleksii Kuchaiev
# The following changes were made:
# LARC support to "optimize_loss" function


#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals

import sys
import collections
from itertools import chain
import six
import re
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

#from open_seq2seq.utils.utils import mask_nans, check_params
from .automatic_loss_scaler import AutomaticLossScaler
from .mp_wrapper import MixedPrecisionOptimizerWrapper

def mask_nans(x):
  x_zeros = tf.zeros_like(x)
  x_mask = tf.is_finite(x)
  y = tf.where(x_mask, x, x_zeros)
  return y

def check_params(config, required_dict, optional_dict):
  if required_dict is None or optional_dict is None:
    return

OPTIMIZER_CLS_NAMES = {
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
    "Ftrl": tf.train.FtrlOptimizer,
    "Momentum": tf.train.MomentumOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "gradients",
    "gradient_norm",
    "global_gradient_norm",
    "variables",
    "variable_norm",
    "larc_summaries",
    "loss_scale"
]


# necessary to redefine this function for pure float16 support
def get_regularization_loss(scope=None, name="total_regularization_loss"):
  """Gets the total regularization loss.

  Args:
    scope: An optional scope name for filtering the losses to return.
    name: The name of the returned tensor.

  Returns:
    A scalar regularization loss.
  """
  losses = tf.losses.get_regularization_losses(scope)
  if losses:
    return tf.add_n(list(map(lambda x: tf.cast(x, tf.float32), losses)),
                    name=name)
  else:
    return tf.constant(0.0)


def reduce_gradients(grads_and_vars, on_horovod, model=None, run_params=None):
  if on_horovod:
    from horovod.tensorflow import allreduce, size
    from horovod.tensorflow.mpi_ops import register_group
    if run_params['hvd_group'] is None :
        layer_indices = get_grads_vars_layer_indices(grads_and_vars, model)
        averaged_grads_and_vars = []
        num_groups = len(layer_indices)
        for idx, layer in enumerate(layer_indices.keys()):
            ind_list = layer_indices[layer]
            if len(ind_list) >= 1:
                layer_grads = [grads_and_vars[ind][0] for ind in ind_list]
                layer_vars = [grads_and_vars[ind][1] for ind in ind_list]
                g_id = register_group(len(layer_grads), "%s:%s:%d" % (layer_grads[0].name, layer_grads[-1].name, idx))  
            if size() > 1:
                avg_grads = [allreduce(grad, compression=run_params['hvd_fp16'], group_id = g_id)
                            if grad is not None else tf.constant(0) for grad in layer_grads ]
                averaged_grads_and_vars.append([(avg_grad, var) for avg_grad, var in zip(avg_grads, layer_vars)])
        print('per layer grouping')        
        return list(chain.from_iterable(averaged_grads_and_vars))
    else:
        num_groups = run_params['hvd_group']
        num_grads_per_group = (len(grads_and_vars) + num_groups - 1) // num_groups
        group_ids = [register_group(num_grads_per_group, "%s:%s:%d" % (grads_and_vars[0][0].name, grads_and_vars[-1][0].name, i))
                for i in range(len(grads_and_vars) // num_grads_per_group)]
    
        if len(grads_and_vars) % num_grads_per_group != 0:
            group_ids.append(register_group(len(grads) % num_grads_per_group, 
                                        "%s:%s:%d" % (grads_and_vars[0][0].name, grads_and_vars[-1][0].name,
                                        len(grads) // num_grads_per_group + 1)))
    if size() > 1:
      averaged_grads_and_vars = []
      with tf.name_scope("all_reduce"):
        for idx, (grad, var) in enumerate(grads_and_vars):
          if grad is not None:
            avg_grad = allreduce(grad, compression=run_params['hvd_fp16'], group_id=group_ids[idx//num_grads_per_group])
            averaged_grads_and_vars.append((avg_grad, var))
          else:
            averaged_grads_and_vars.append((tf.constant(0), var))
      return averaged_grads_and_vars
    else:
      return grads_and_vars
  else:
    raise NotImplementedError("Reduce in tower-mode is not implemented.")


def optimize_loss(loss,
                  optimizer,
                  optimizer_params,
                  learning_rate_decay_fn,
                  run_params=None,
                  var_list=None,
                  dtype=tf.float32,
                  clip_gradients=None,
                  summaries=None,
                  hyper_params=None,
                  loss_scaling=1.0,
                  loss_scaling_params=None,
                  on_horovod=False,
                  iter_size=1,
                  skip_update_cond=None,
                  model=None,
                  model_scopes=None):
  """Given loss and parameters for optimizer, returns a training op.

  Args:
    loss: Scalar `Tensor`.
    optimizer: string or class of optimizer, used as trainer.
        string should be name of optimizer, like 'SGD',
        'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
        class should be sub-class of `tf.Optimizer` that implements
        `compute_gradients` and `apply_gradients` functions.
    optimizer_params: parameters of the optimizer.
    var_list: List of trainable variables. Can be used to freeze
        certain trainable variables by excluding them from this list. 
        If set to None, all trainable variables will be optimized.
    dtype: model dtype (tf.float16, tf.float32 or "mixed").
    learning_rate_decay_fn: function, takes `global_step`
        `Tensor`s, returns `Tensor`.
        Can be used to implement any learning rate decay
        functions.
        For example: `tf.train.exponential_decay`.
        Ignored if `learning_rate` is not supplied.
    clip_gradients: float, max gradient norm to clip to.
    summaries: List of internal quantities to visualize on tensorboard. If not
        set only the loss and the learning rate will be reported. The
        complete list is in OPTIMIZER_SUMMARIES.
    hyper_params: If not None, gradient re-scaling will
        be applied with corresponding methods/parameters.
    loss_scaling: could be float or string. If float, static loss scaling
        is applied. If string, the corresponding automatic
        loss scaling algorithm is used. Must be one of 'Backoff'
        of 'LogMax' (case insensitive). Only used when dtype="mixed".
    on_horovod: whether the model is run on horovod.

  Returns:
    training op.
  """
  if summaries is None:
    summaries = ["learning_rate", "global_gradient_norm", "loss_scale"]
  else:
    for summ in summaries:
      if summ not in OPTIMIZER_SUMMARIES:
        raise ValueError(
            "Summaries should be one of [{}], you provided {}.".format(
                ", ".join(OPTIMIZER_SUMMARIES), summ,
            )
        )
  if clip_gradients is not None and hyper_params['LARC']:
    raise AttributeError(
        "LARC and gradient norm clipping should not be used together"
    )
    
  if run_params['grad_ckpt'] is not None: 
    from . import memory_saving_gradients
    from tensorflow.python.ops import gradients
    def gradients_memory(ys, xs, grad_ys=None, **kwargs):
      return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints=run_params['grad_ckpt'], **kwargs)
    gradients.__dict__["gradients"] = gradients_memory 

  global_step = tf.train.get_or_create_global_step()
  lr = learning_rate_decay_fn(global_step)
  if "learning_rate" in summaries:
    tf.summary.scalar("learning_rate", lr)

  with tf.variable_scope("Loss_Optimization"):
    update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    loss = control_flow_ops.with_dependencies(list(update_ops), loss)

    # Create optimizer, given specified parameters.
    if isinstance(optimizer, six.string_types):
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError(
            "Optimizer name should be one of [{}], you provided {}.".format(
                ", ".join(OPTIMIZER_CLS_NAMES), optimizer
            )
        )
      optimizer = OPTIMIZER_CLS_NAMES[optimizer]
    opt = optimizer(learning_rate=lr, **optimizer_params)

    if isinstance(loss_scaling, six.string_types):
      loss_scaling = AutomaticLossScaler(
          algorithm=loss_scaling,
          params=loss_scaling_params
      )
      if "loss_scale" in summaries:
        tf.summary.scalar("loss_scale", loss_scaling.loss_scale)
    else:
        loss_scaling=None
    if dtype == 'mixed':
      opt = MixedPrecisionOptimizerWrapper(opt, loss_scale=loss_scaling)

    # Compute gradients.
    grads_and_vars = opt.compute_gradients(
        loss, colocate_gradients_with_ops=True, var_list=var_list
    )

    if on_horovod:
      if iter_size > 1 :
        grads_and_vars_accum = []
        accum_ops = []
        for grad, var in grads_and_vars:
          # necessary to use tf.Variable directly to instantiate cudnn rnn cells
          # which don't have explicit shape.
          grad_accum = tf.Variable(
              initial_value=tf.zeros_like(var),
              name=grad.name.split(":")[0] + "_accum",
              expected_shape=var.shape,
              dtype=grad.dtype,
              trainable=False,
              validate_shape=bool(var.get_shape())
          )
          if isinstance(grad, tf.IndexedSlices):
            add_grads = tf.scatter_nd_add(grad_accum, grad.indices,
                                          grad.values / iter_size)
          else:
            add_grads = grad_accum + grad / iter_size

          accum_ops.append(tf.assign(grad_accum, add_grads))
          grads_and_vars_accum.append((grad_accum, var))
        
        def accumulate():
            return tf.group(accum_ops)
 
        #accum_op = tf.group(accum_ops) 

        def update_and_clear_op():
          with tf.control_dependencies([accumulate()]):
            red_grad_updates = opt.apply_gradients(
                post_process_gradients(
                    reduce_gradients(grads_and_vars_accum, on_horovod=True, model=model_scopes, run_params=run_params),
                    lr=lr,
                    clip_gradients=clip_gradients,
                    hyper_params=hyper_params,
                    summaries=summaries,
                    model_scopes=model_scopes
                ),
                global_step=None,
            )
          with tf.control_dependencies([red_grad_updates]):
            update_ops = tf.group([tf.assign(g, tf.zeros_like(g))
                            for g, v in grads_and_vars_accum])
          return update_ops

        grad_updates = tf.cond(
            pred=skip_update_cond,
            false_fn=update_and_clear_op, 
            true_fn=accumulate
        )
      else:
        grad_updates = opt.apply_gradients(
            post_process_gradients(
                reduce_gradients(grads_and_vars, on_horovod=True, model=model_scopes, run_params=run_params),
                lr=lr,
                clip_gradients=clip_gradients,
                hyper_params=hyper_params,
                summaries=summaries,
                model_scopes=model_scopes
            ),
            global_step=None
        )
    else:
      grad_updates = opt.apply_gradients(
          post_process_gradients(
              grads_and_vars,
              lr=lr,
              clip_gradients=clip_gradients,
              hyper_params=hyper_params,
              summaries=summaries,
              model_scopes=model_scopes
          ),
          global_step=None,
      )

    # Ensure the train_tensor computes grad_updates.
    train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)

    return train_tensor, lr


def post_process_gradients(grads_and_vars, summaries, lr,
                           clip_gradients, hyper_params, model_scopes=None):
  """Applies post processing to gradients, i.e. clipping, LARC, summaries."""
  if "global_gradient_norm" in summaries:
    tf.summary.scalar(
        "global_gradient_norm",
        _global_norm_with_cast(grads_and_vars),
    )

  # Optionally clip gradients by global norm.
  if clip_gradients is not None:
    grads_and_vars = _clip_gradients_by_norm(grads_and_vars, clip_gradients)

  # Add histograms for variables, gradients and gradient norms.
  for gradient, variable in grads_and_vars:
    if isinstance(gradient, tf.IndexedSlices):
      grad_values = gradient.values
    else:
      grad_values = gradient

    if isinstance(variable, tf.IndexedSlices):
      var_values = variable.values
    else:
      var_values = variable

    if grad_values is not None:
      var_name = variable.name.replace(":", "_")
      if "gradients" in summaries:
        # need to mask nans for automatic loss scaling
        tf.summary.histogram("gradients/%s" % var_name, mask_nans(grad_values))
      if "gradient_norm" in summaries:
        tf.summary.scalar("gradient_norm/%s" % var_name, tf.norm(grad_values))
      if "variables" in summaries:
        tf.summary.histogram("variables/%s" % var_name, var_values)
      if "variable_norm" in summaries:
        tf.summary.scalar("variable_norm/%s" % var_name, tf.norm(var_values))

  if clip_gradients is not None and "global_gradient_norm" in summaries:
    tf.summary.scalar(
        "global_clipped_gradient_norm",
        _global_norm_with_cast(grads_and_vars),
    )

  # re-scaling gradients using some specified method
  # 1. get layer index dictionary for each gradient/var 
  if model_scopes is not None:
    layer_indices = get_grads_vars_layer_indices(grads_and_vars, model_scopes)
    new_grads_vars = []
    for layer in layer_indices.keys():
      ind_list = layer_indices[layer]
      if len(ind_list) >= 1:
          layer_grads = [grads_and_vars[ind][0] for ind in ind_list]
          layer_vars = [grads_and_vars[ind][1] for ind in ind_list]
          grad_vec = tf.concat([tf.expand_dims(tf.reshape(grad, [-1]), 0) for grad in layer_grads], 1)
          var_vec = tf.concat([tf.expand_dims(tf.reshape(var, [-1]), 0) for var in layer_vars], 1)
          var_dtype = layer_vars[0].dtype
          var_nom = tf.norm(tensor=tf.cast(var_vec, tf.float32))
          grad_norm = tf.norm(tensor=tf.cast(grad_vec, tf.float32))
          if hyper_params['LARC']:
            check_params( config=hyper_params,
                          required_dict={'LARC_eta': float},
                          optional_dict={
                              'LARC_mode': ['clip', 'scale'],
                              'LARC_min_update': float,
                              'LARC_epsilon': float
                          },
                          )
            larc_eta = hyper_params['LARC_eta']
            larc_mode = hyper_params.get('LARC_mode', 'clip')
            min_update = hyper_params.get('LARC_min_update', 1e-7)
            eps = hyper_params.get('LARC_epsilon', 1e-7)

            if larc_mode == 'scale':
              grad_updates = [ tf.minimum( tf.maximum( larc_eta * var_nom / (grad_norm + eps), min_update), 1) * grad 
                                for grad in layer_grads]
            elif larc_mode == 'clip':
              grad_updates = [ tf.minimum( tf.maximum( larc_eta * var_nom / ( lr * (grad_norm + eps)), min_update), 1) * grad 
                                for grad in layer_grads]
          elif hyper_params['LSAL']:
            check_params( config=hyper_params,
                          required_dict={},
                          optional_dict={
                              'LSAL_min_update': float,
                              'LSAL_epsilon': float
                          },
                          )
            min_update = hyper_params.get('LSAL_min_update', 1e-7)
            eps = hyper_params.get('LSAL_epsilon', 1e-7)  
            grad_updates = [ tf.maximum( 1 + tf.log1p( 1/ (grad_norm + eps)), min_update) * grad 
                              for grad in layer_grads]
          else:
             grad_updates = layer_grads
          new_grads_vars_layer = [( tf.saturate_cast(grad_update, var_dtype ), var) 
                                      for grad_update, var in zip(grad_updates, layer_vars)]      
          new_grads_vars.append(new_grads_vars_layer)
    new_grads_vars = list(chain.from_iterable(new_grads_vars))
    return new_grads_vars
  else:
    return grads_and_vars


def get_grads_vars_layer_indices(grads_vars, scopes):
    ind_dict = collections.OrderedDict()
    for scope in scopes:
        p = re.compile(scope.name)
        ind_list = []
        for (ind, grad) in enumerate(grads_vars):
            if p.search(grad[1].name):
                ind_list.append(ind)
        ind_dict[scope.name] = ind_list
    return ind_dict

## TODO: add gradient update summaries effective_lr tensor 
      # if "larc_summaries" in summaries:
      #     tf.summary.scalar('larc_clip_on/{}'.format(v.name),
      #                       tf.cast(tf.less(larc_grad_update, 1.0), tf.int32))

      # # adding additional summary
      # if "larc_summaries" in summaries:
      #   tf.summary.scalar('larc_grad_update/{}'.format(v.name),
      #                     larc_grad_update)
      #   tf.summary.scalar("larc_final_lr/{}".format(v.name),
      #                     tf.cast(lr, var_dtype) * larc_grad_update)

def _global_norm_with_cast(grads_and_vars):
  return tf.global_norm(list(map(
      lambda x: tf.cast(x, tf.float32),
      list(zip(*grads_and_vars))[0]
  )))


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  dtypes = [var.dtype for var in variables]

  # Clip gradients in float32
  clipped_gradients, _ = _clip_by_global_norm(
      gradients,
      clip_gradients,
      use_norm=_global_norm_with_cast(grads_and_vars)
  )

  # Convert gradients back to the proper dtype
  clipped_gradients = [
      tf.cast(grad, dtype)
      for grad, dtype in zip(clipped_gradients, dtypes)
  ]

  return list(zip(clipped_gradients, variables))


def _clip_by_global_norm(t_list, clip_norm, use_norm, name=None):
  """Clips values of multiple tensors by the ratio of the sum of their norms.
  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. The global
  norm is expected to be pre-computed and passed as use_norm.
  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
  Any of the entries of `t_list` that are of type `None` are ignored.
  This is the correct way to perform gradient clipping (for example, see
  [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).
  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).

  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.

  Raises:
    TypeError: If `t_list` is not a sequence.
  """
  if (not isinstance(t_list, collections.Sequence)
      or isinstance(t_list, six.string_types)):
    raise TypeError("t_list should be a sequence")
  t_list = list(t_list)

  # Removed as use_norm should always be passed
  # if use_norm is None:
  #   use_norm = global_norm(t_list, name)

  with tf.name_scope(name, "clip_by_global_norm",
                     t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale = clip_norm * tf.minimum(
        1.0 / use_norm,
        tf.ones([1], dtype=use_norm.dtype) / clip_norm)

    values = [
        tf.cast(
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i),
            dtype=tf.float32
        )
        if t is not None else t
        for i, t in enumerate(t_list)]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with tf.colocate_with(v):
          values_clipped.append(
              tf.identity(v * scale, name="%s_%d" % (name, i)))

    list_clipped = [
        tf.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, tf.IndexedSlices)
        else c_v
        for (c_v, t) in zip(values_clipped, t_list)]

  return list_clipped, use_norm
  
