# encoding=utf-8

import tensorflow as tf
import six
import json
import copy
from utils.transformer import transformer
from utils.embedding_util import embedding_lookup, embedding_postprocessor
from utils.misc_utils import get_shape_list, create_2sent_3d_attention_mask, get_activation, build_rnn


class SentConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=512,
                 num_layers=8,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 max_position_embeddings=128,
                 initializer_range=0.02,
                 num_sents=2,
                 max_sent_length=64,
                 use_rnn=True,
                 num_rnn_layers=2,
                 rnn_dir='uni'):
        """

        :param vocab_size:
        :param hidden_size:
        :param num_layers:
        :param num_attention_heads:
        :param intermediate_size:
        :param hidden_act:
        :param hidden_dropout_prob:
        :param attention_dropout_prob:
        :param max_position_embeddings:
        :param initializer_range:
        """

        self.vocab_size = vocab_size,
        self.hidden_size = hidden_size,
        self.num_layers = num_layers,
        self.num_attention_heads = num_attention_heads,
        self.intermediate_size = intermediate_size,
        self.hidden_act = hidden_act,
        self.hidden_dropout_prob = hidden_dropout_prob,
        self.attention_dropout_prob = attention_dropout_prob,
        self.max_position_embeddings = max_position_embeddings,
        self.initializer_range = initializer_range
        self.num_sents = num_sents
        self.max_sent_length = max_sent_length
        self.use_rnn = use_rnn
        self.num_rnn_layers = num_rnn_layers
        self.rnn_dir = rnn_dir

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = SentConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentModel(object):
    """
    Input: vocab ids corresponding to natural language sentences.

    input_ids = tf.constant([batch_size, seq_length])

    sents_length: [B, 2], length of each sentences
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 sents_length,
                 input_mask=None,
                 token_type_ids=None,
                 scope=None,):

        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size, seq_length = input_shape

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name='Sent'):
            with tf.variable_scope('embeddings'):
                self.embedding_output, self.embedding_table = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name='word_embeddings')

                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=False,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    use_sent_position_embeddings=True,
                    num_sents=config.num_sents,
                    max_sent_length=config.max_sent_length,
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope('encoder'):
                attention_mask = create_2sent_3d_attention_mask(self.embedding_output, input_mask)

                self.all_encoder_layers = transformer(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            # self.sequence_output: [B, S, E]
            self.sequence_output = self.all_encoder_layers[-1]

            with tf.variable_scope('RNN'):
                # [B, S/2, E]
                self.sent1_output, self.sent2_output = tf.split(
                    self.sequence_output, num_or_size_splits=2, axis=1)

                with tf.variable_scope('dot_product'):
                    self.dot_product = self.sent1_output * self.sent2_output

                with tf.variable_scope('rnn_output'):

                    # after dot product, dynamic rnn only needs to process min_length of seq
                    min_length = tf.reduce_min(sents_length, axis=-1)

                    # [B, S, E]
                    self.rnn_output, self.rnn_state = build_rnn(
                        self.dot_product,
                        config.hidden_size,
                        config.num_rnn_layers,
                        config.hidden_dropout_prob,
                        config.rnn_dir,
                        min_length)

                    # pick the last relevant output of RNN
                    batch_range = tf.range(tf.shape(self.rnn_output)[0])
                    indices = tf.stack([batch_range, min_length - 1], axis=1)
                    self.final_output = tf.gather_nd(self.rnn_output, indices)

    def get_output(self):
        # [B, E]
        return self.final_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table
