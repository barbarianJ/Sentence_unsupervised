# encoding=utf-8

from model.model import SentConfig, SentModel
import os
import codecs
from tqdm import tqdm
from hp import *
import tensorflow as tf
from utils import jieba_tokenization as tokenization, optimization
# from gdu import gdu
from random import shuffle, sample
from utils.misc_utils import get_assignment_map_from_checkpoint


class Classifier(object):

    def __init__(self, model_config, num_labels, batch_size, num_train_steps=None, is_training=True):
        self.input_ids = tf.placeholder(shape=(batch_size, max_seq_length), dtype=tf.int32, name='input_ids')
        # self.input_mask = tf.placeholder(shape=(batch_size, 2, sent_length), dtype=tf.int32, name='input_mask')
        # self.input_sents_length = tf.placeholder(shape=(batch_size, 2), dtype=tf.int32, name='segment_ids')
        self.input_mask = tf.placeholder(shape=(batch_size, max_seq_length), dtype=tf.int32, name='input_mask')
        self.segment_ids = tf.placeholder(shape=(batch_size, max_seq_length), dtype=tf.int32, name='segment_ids')
        self.label_id = tf.placeholder(shape=(batch_size, 1), dtype=tf.int32, name='label_id')

        model = SentModel(
            config=model_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            # sents_length=self.input_sents_length
            token_type_ids=self.segment_ids
        )

        model_output = model.get_output()
        hidden_size = model_output.shape[-1].value

        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                model_output = tf.nn.dropout(model_output, keep_prob=0.9)

            logits = tf.matmul(model_output, output_weights, transpose_b=True)
            self.logits = tf.nn.bias_add(logits, output_bias)
            self.probabilities = tf.nn.softmax(logits, axis=-1)

            if is_training:
                # self.label_id = tf.placeholder(shape=(batch_size, 1), dtype=tf.int32, name='label_id')

                log_probs = tf.nn.log_softmax(logits, axis=-1)

                one_hot_labels = tf.one_hot(self.label_id, depth=num_labels, dtype=tf.float32)

                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)

                num_warmup_steps = int(num_warmup_proportion * num_train_steps)
                self.train_op, self.global_step = optimization.create_optimizer(self.loss,
                                                                                learning_rate, num_train_steps,
                                                                                num_warmup_steps, use_tpu=False)

                # saver
                self.saver = tf.train.Saver(var_list=tf.trainable_variables() + [self.global_step],
                                            max_to_keep=5)
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()
                self.summary_writter = tf.summary.FileWriter(os.path.join(output_dir, 'train_summary'),
                                                             tf.get_default_graph())

    def train(self, sess, feed_values, summary=False, saver=False):
        sess.run(self.train_op, feed_dict=self.make_feed_dict(*feed_values))

        if summary:
            summary = sess.run(self.summary_op,
                               feed_dict=self.make_feed_dict(*feed_values))
            self.summary_writter.add_summary(summary, self.global_step.eval(session=sess))

        if saver:
            self.saver.save(sess, output_dir + '/ckpt', global_step=self.global_step)

    def infer(self, sess, feed_values):
        return sess.run(self.probabilities, feed_dict=self.make_feed_dict(*feed_values))

    def restore_model(self, sess):
        latest_ckpt = tf.train.latest_checkpoint(output_dir)
        self.saver.restore(sess, latest_ckpt)

    def restore_ckpt_global_step(self, init_checkpoint=None, include_global_step=True):
        tf.global_variables_initializer().run()

        tvars = tf.trainable_variables()
        if include_global_step:
            tvars += [self.global_step]
        initialized_variable_names = {}

        if init_checkpoint:
            assignment_map, initialized_variable_names = \
                get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = %s, shape = %s%s", var.name, var.shape,
                  init_string)

    def make_feed_dict(self, ids, masks, segment_ids, labels):
        return {
            self.input_ids: ids,
            self.input_mask: masks,
            # self.input_sents_length: sents_length,
            self.segment_ids: segment_ids,
            self.label_id: labels}


class DataProcessor(object):

    # data format: 'sent1 && sent2 && label'
    def __init__(self, sent_length, tokenizer, true_file, false_file, infer_file, max_seq_length):
        self.sent_length = sent_length
        self.tokenizer = tokenizer
        self.train_data = []
        self.infer_data = []
        self.next_idx = 0
        self.train_true_file = true_file
        self.train_false_file = false_file
        self.infer_file = infer_file
        self.max_seq_length = max_seq_length

    def prepare_train_data(self):
        with codecs.open(self.train_true_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not (line.startswith('&& ') or line.endswith(' &&')):
                    self.train_data.append(line + ' && ' + '0')

        with codecs.open(self.train_false_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not (line.startswith('&& ') or line.endswith(' &&')):
                    self.train_data.append(line + ' && ' + '1')
        shuffle(self.train_data)

    def prepare_infer_data(self):
        with codecs.open(self.infer_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.infer_data.append(line.strip())

        return self.infer_data

    def get_num_train_data(self):
        return len(self.train_data)

    def get_num_infer_data(self):
        return len(self.infer_data)

    def get_train_data(self, batch_size):
        return self._create_train_data(batch_size)

    def create_infer_data(self, sent1, sent2):
        ids = []
        masks = []
        sents_length = []
        labels = [[0]]

        ids_a = self._text_to_ids(sent1)
        ids_b = self._text_to_ids(sent2)

        mask_a = [1] * len(ids_a)
        mask_b = [1] * len(ids_b)

        # generate needed data
        sent_length = [len(ids_a), len(ids_b)]

        ids_a, mask_a = self._pad_id_mask(ids_a, mask_a)
        ids_b, mask_b = self._pad_id_mask(ids_b, mask_b)

        mask = [mask_a, mask_b]

        ids.append(ids_a + ids_b)
        masks.append(mask)
        sents_length.append(sent_length)

        return ids, masks, sents_length, labels

    def _pad_id_mask(self, input_ids, input_mask):
        while len(input_ids) < self.sent_length:
            input_ids.append(0)
            input_mask.append(0)

        return input_ids, input_mask

    def _text_to_ids(self, input_text):
        text = tokenization.convert_to_unicode(input_text)
        token = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(token)

        return ids

    def _create_train_data(self, batch_size):
        ids = []
        masks = []
        sents_length = []
        labels = []

        if batch_size + self.next_idx > len(self.train_data):
            self.next_idx = 0
            shuffle(self.train_data)
        data = self.train_data[self.next_idx: self.next_idx + batch_size]
        self.next_idx += batch_size

        for d in data:
            text_a, text_b, label = d.split(' && ')

            label = int(label)

            ids_a = self._text_to_ids(text_a)
            ids_b = self._text_to_ids(text_b)

            mask_a = [1] * len(ids_a)
            mask_b = [1] * len(ids_b)

            # generate needed data
            sent_length = [len(ids_a), len(ids_b)]

            ids_a, mask_a = self._pad_id_mask(ids_a, mask_a)
            ids_b, mask_b = self._pad_id_mask(ids_b, mask_b)

            mask = [mask_a, mask_b]

            ids.append(ids_a + ids_b)
            masks.append(mask)
            sents_length.append(sent_length)
            labels.append([label])

        # id: [B, S]
        # masks: [B, 2, sent_length]
        # sents_length: [B, 2]
        # labels" [B, 1]
        return ids, masks, sents_length, labels

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _text_to_ids_v2(self, text_a, max_seq_length, text_b=None):
        text_a = tokenization.convert_to_unicode(text_a)
        token1 = self.tokenizer.tokenize(text_a)

        token2 = None
        if text_b:
            text_b = tokenization.convert_to_unicode(text_b)
            token2 = self.tokenizer.tokenize(text_b)

            self._truncate_seq_pair(token1, token2, max_seq_length - 3)
        else:
            token1 = token1[:max_seq_length - 2]

        # format input data
        tokens = ['[CLS]'] + token1 + ['[SEP]']
        segment_ids = [0] * len(tokens)

        if token2:
            tokens += token2 + ['[SEP]']
            segment_ids += [1] * (len(token2) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        padding_length = max_seq_length - len(input_ids)
        padding_value = [0] * padding_length
        input_ids += padding_value
        input_mask += padding_value
        segment_ids += padding_value

        return input_ids, input_mask, segment_ids

    def get_train_data_v2(self, batch_size):
        input_ids = []
        input_masks = []
        segment_ids = []
        label_ids = []

        if batch_size + self.next_idx > len(self.train_data):
            self.next_idx = 0
            shuffle(self.train_data)
        data = self.train_data[self.next_idx: self.next_idx + batch_size]
        self.next_idx += batch_size

        for d in data:
            text_a, text_b, label = d.split(' && ')

            ids, masks, seg_ids = \
                self._text_to_ids_v2(text_a=text_a, max_seq_length=max_seq_length, text_b=text_b)

            input_ids.append(ids)
            input_masks.append(masks)
            segment_ids.append(seg_ids)
            label_ids.append([int(label)])

        return input_ids, input_masks, segment_ids, label_ids

    def create_infer_data_v2(self, sent1, sent2):
        ids, masks, seg_ids = \
            self._text_to_ids_v2(text_a=sent1, max_seq_length=max_seq_length, text_b=sent2)

        label = [[0]]
        return [ids], [masks], [seg_ids], label


def main():
    model_config = SentConfig.from_json_file(model_config_file)

    if max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the model "
            "was only trained up to sequence length %d" %
            (max_seq_length, model_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)
    tf.gfile.MakeDirs(infer_output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    processor = DataProcessor(sent_length=sent_length, tokenizer=tokenizer,
                              true_file=true_file, false_file=false_file,
                              infer_file=infer_file, max_seq_length=max_seq_length)

    if train:
        processor.prepare_train_data()
        num_data = processor.get_num_train_data()
        num_train_steps = num_epoch * num_data // batch_size

        with tf.Graph().as_default() as global_graph:

            model = Classifier(model_config, num_labels=2, batch_size=batch_size,
                               num_train_steps=num_train_steps, is_training=True)

            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(graph=global_graph, config=config) as sess:
                # model.restore_ckpt_global_step(init_checkpoint=init_checkpoint, include_global_step=True)
                model.restore_model(sess)

                tq = tqdm(range(1, num_train_steps + 1))
                for step in tq:
                    ids, masks, sents_length, labels = processor.get_train_data_v2(batch_size)

                    model.train(sess, (ids, masks, sents_length, labels), summary=not step % 100, saver=not step % 1000)

    elif infer:

        with tf.Graph().as_default() as global_graph:

            model = Classifier(model_config, num_labels=2, batch_size=batch_size, is_training=False)

            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(graph=global_graph, config=config) as sess:
                # model.restore_ckpt_global_step(init_checkpoint=init_checkpoint,include_global_step=False)
                model.restore_model(sess)

                infer_data = processor.prepare_infer_data()
                length = len(infer_data)
                match_found = False

                tf.train.write_graph(sess.graph_def, 'result/', 'model_graph.pbtxt')
                for sent1_idx in tqdm(range(infer_start_index, length)):
                    for sent2_idx in tqdm(sample(range(length), min(length, num_sent_to_compare))):
                        if match_found:
                            match_found = False
                            break

                        if sent1_idx == sent2_idx or infer_data[sent1_idx] == infer_data[sent2_idx]:
                            continue

                        if len(infer_data[sent2_idx]) < 3 or len(infer_data[sent2_idx]) > 20:
                            continue

                        if len(infer_data[sent2_idx]) > 5 * len(infer_data[sent1_idx]):
                            continue

                        ids, masks, sents_length, labels = \
                            processor.create_infer_data_v2(infer_data[sent1_idx], infer_data[sent2_idx])

                        prob = model.infer(sess, (ids, masks, sents_length, labels))

                        prob = prob[0][0]
                        if infer_lower_bound < prob < infer_upper_bound:
                            output_file = os.path.join(infer_output_dir, 'predictions.tsv')
                            with tf.gfile.GFile(output_file, "a") as writer:
                                result_line = ' && '.join(
                                    [infer_data[sent1_idx], infer_data[sent2_idx], str(prob)]
                                ) + '\n'

                                writer.write(result_line)

                            match_found = True


if __name__ == "__main__":
    main()
