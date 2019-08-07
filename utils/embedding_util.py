# encoding=utf-8

from .misc_utils import *


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name='word_embeddings',
                     use_one_hot_embeddings=False):
    """

    :param input_ids:
    :param vocab_size:
    :param embedding_size:
    :param initializer_range:
    :param word_embedding_name:
    :param use_one_hot_embeddings:
    :return: float, [B, S, E]
    """

    # add dim -> [B, S, 1], first two dims will be used in input_shape to reshape output
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[:-1] + [input_shape[-1] * embedding_size])

    return output, embedding_table


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=2,
                            token_type_embedding_name='token_type_embeddings',
                            use_position_embeddings=True,
                            position_embedding_name='position_embeddings',
                            use_sent_position_embeddings=True,
                            num_sents=2,
                            sent_length=None,
                            initializer_range=0.02,
                            max_position_embeddings=128,
                            dropout_prob=0.1):
    """

    :param input_tensor: consists of one or many sub sentence component, separated by [SEP], shape [B, S, E]
    :param use_token_type: if more than one sub sentence component, each sub sent has different token_type_id
    :param token_type_ids: segment ids, corresponding to different sub sentence component
                            shape [B, S]
    :param token_type_vocab_size:
    :param token_type_embedding_name:
    :param use_position_embeddings:
    :param position_embedding_name:
    :param use_sent_position_embeddings: whether to use position embedding within each sub sentence component,
                                or use one position embedding for the whole input_tensor
    :param num_sents: number of sub sentence components
    :param sent_length: int list, [B, S], length of each sent
    :param initializer_range:
    :param max_position_embeddings:
    :param dropout_prob:
    :return: float, [B, S ,E]
    """

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size, seq_length, width = input_shape

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("'token_type_ids' must be specified if use_token_type")

        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))

        flat_token_type_ids = tf.reshape(token_type_ids, [-1])

        print('use tf.gather instead of one hot')
        # one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        # token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)

        token_type_embeddings = tf.gather(token_type_table, flat_token_type_ids)
        token_type_embeddings = tf.reshape(token_type_embeddings, input_shape)

        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            if use_sent_position_embeddings:
                if not sent_length:
                    raise ValueError("'sent_length' must be specified")

                if not max_position_embeddings % num_sents == 0:
                    raise ValueError("'max_position_embeddings' must be multiple of 'num_sents'"
                                     "got max_position_embeddings: %d & num_sents: %d"
                                     % (max_position_embeddings, num_sents))

                full_position_embeddings = tf.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings // num_sents, width],
                    initializer=create_initializer(initializer_range))

                # slice position_embedding for each sub sent
                sent_position_embeddings = []
                for sent_idx in range(num_sents):
                    sent_position_embeddings.append(
                        tf.slice(full_position_embeddings, [0, 0],
                                 [sent_length[sent_idx], width]))

                position_embeddings = tf.concat(sent_position_embeddings, axis=0)

            else:
                full_position_embeddings = tf.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings, width],
                    initializer=create_initializer(initializer_range))

                position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                               [seq_length, -1])

            num_dims = len(output.shape.as_list())

            position_broadcast_shape = []
            for _ in range(num_dims):
                position_broadcast_shape.append(1)

            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings

        output = layer_norm(output)
        output = dropout(output, dropout_prob)

        return output
