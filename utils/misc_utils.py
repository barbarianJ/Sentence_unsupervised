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


def create_2sent_3d_attention_mask(from_tensor, to_mask):
    """
    This function assumes that from_tensor and to_tensor have the same seq_length.
    :param from_tensor: 2D or 3D Tensor, [B, S, ..]
    :param to_mask: 3D Tensor, [B, num_sents, to_seq_length]
    :return:
        float Tensor of shape [B, from_seq_length, to_seq_length]
    """
    mask1 = _create_2sent_3d_attention_mask(from_tensor, to_mask[:, 0, :], 0)
    mask2 = _create_2sent_3d_attention_mask(from_tensor, to_mask[:, 1, :], 1)

    mask = tf.concat((mask1, mask2), axis=-2)

    return mask


def _create_2sent_3d_attention_mask(from_tensor, to_mask, sent_number):
    """

    :param from_tensor: 2D or 3D Tensor, [B, S, ..]
    :param to_mask: 2D Tensor, [B, to_seq_length]
    :param sent_number:
    :return:
        float Tensor of shape [B, from_seq_length, to_seq_length]
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size, from_seq_length = from_shape[:2]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    mask_padding = tf.zeros(to_shape, dtype=tf.float32)
    to_mask = tf.cast(to_mask, tf.float32)

    if sent_number == 0:
        to_mask = tf.concat((to_mask, mask_padding), axis=-1)
    else:
        to_mask = tf.concat((mask_padding, to_mask), axis=-1)

    to_mask = tf.reshape(to_mask, [batch_size, 1, -1])

    broadcast_ones = tf.ones(
        shape=[batch_size, to_seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * to_mask

    return mask


def test_mask():
    inputs = tf.range(16)
    inputs = tf.reshape(inputs, [2, 8], name='inputs')

    mask = tf.constant(
        [
            [[1, 1, 0, 0], [1, 0, 0, 0]],
            [[1, 0, 1, 0], [0, 0, 1, 0]]
        ], dtype=tf.float32
    )

    # tf.reset_default_graph()
    # with tf.get_default_graph():
    with tf.Session() as sess:

        res = sess.run([create_2sent_3d_attention_mask(inputs, mask)])

        print(res)

