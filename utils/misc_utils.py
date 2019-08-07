# encoding=utf-8

import tensorflow as tf
import numpy as np
import six


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_shape_list(tensor, expected_rank=None, name=None):
    """

    :param tensor:
    :param expected_rank:
    :param name:
    :return:
        list of dimensions of shape of tensor. dynamic dimensions will be returned as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()
    dynamic_index = []
    for idx, dim in enumerate(shape):
        if dim is None:
            dynamic_index.append(idx)

    if not dynamic_index:
        return shape

    dynamic_shape = tf.shape(tensor)
    for idx in dynamic_index:
        shape[idx] = dynamic_shape[idx]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """
    assert tensor.rank == expected_rank

    :param tensor:
    :param expected_rank:
    :param name:
    :return:
    """

    if name is None:
        name = tensor.name

    if isinstance(expected_rank, six.integer_types):
        expected_rank = [expected_rank]

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank:
        scope_name = tensor.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def reshape_to_matrix(input_tensor):
    """reshape 3D to 2D, [B, S, E] -> [B * S, E]"""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError('## reshape_to_matrix: input tensor must have at least rank 2. Shape = %s' %
                         input_tensor.shape)

    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    input_tensor = tf.reshape(input_tensor, [-1, width])
    return input_tensor


def reshape_from_matrix(input_tensor, to_shape):
    """reshape from 2D to 3D"""
    if len(to_shape) == 2:
        return input_tensor

    output_shape = get_shape_list(input_tensor)

    to_dim = to_shape[:-1]
    width = output_shape[-1]

    return tf.reshape(input_tensor, to_dim + [width])


def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(tensor, prob):
    return tf.nn.dropout(tensor, 1.0 - prob)


def layer_norm(tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
