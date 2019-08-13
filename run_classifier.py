# encoding=utf-8

from model.model import SentConfig, SentModel
import os
import codecs
from tqdm import tqdm
from hp import *
import tensorflow as tf
from utils import jieba_tokenization as tokenization, optimization
from gdu import gdu
from random import shuffle, sample
from utils.misc_utils import get_assignment_map_from_checkpoint


model_config_file = 'model/model_config/config.json'
max_seq_length = 100
output_dir = 'result/'
train = True
infer = False
vocab_file = 'model/model_config/vocab.txt'
do_lower_case = False
true_file = 'data/handwritten_qingyun/han_qing_true.txt'
false_file = 'data/handwritten_qingyun/han_qing_false.txt'

infer_file = 'data/crawled/crawled.txt'
infer_output_dir = 'infer/'

init_checkpoint = None

batch_size = 64
sent_length = max_seq_length // 2
num_epoch = 10000
learning_rate = 0.00005
num_warmup_proportion = 0.1

infer_start_index = 0
num_sent_to_compare = 100000

infer_lower_bound = -0.05
infer_upper_bound = 0.05


def create_model(model_config, is_training, input_ids, sents_length, input_mask,
                 labels):
    model = SentModel(
        config=model_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        sents_length=sents_length
    )

    model_output = model.get_output()
    print('model_output shape: ' + str(model_output.shape))

    with tf.variable_scope('loss'):
        predict = tf.layers.dense(model_output, 1,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                  bias_initializer=tf.zeros_initializer(),
                                  activation=tf.sigmoid)

        loss = tf.losses.mean_squared_error(predict, labels)

    return loss, model_output


class DataProcessor(object):

    # data format: 'sent1 && sent2 && label'
    def __init__(self, sent_length, tokenizer, true_file, false_file, infer_file):
        self.sent_length = sent_length
        self.tokenizer = tokenizer
        self.train_data = []
        self.infer_data = []
        self.next_idx = 0
        self.train_true_file = true_file
        self.train_false_file = false_file
        self.infer_file = infer_file

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
        labels = [0]

        ids_a = self._text_to_ids(sent1)
        ids_b = self._text_to_ids(sent2)

        mask_a = [1] * len(ids_a)
        mask_b = [1] * len(ids_b)

        # generate needed data
        sent_length = [len(ids_a), len(ids_b)]

        ids_a, mask_a = self._pad_id_seq(ids_a, mask_a)
        ids_b, mask_b = self._pad_id_seq(ids_b, mask_b)

        mask = [mask_a, mask_b]

        ids.append(ids_a + ids_b)
        masks.append(mask)
        sents_length.append(sent_length)

        return ids, masks, sents_length, labels

    def _pad_id_seq(self, input_ids, input_mask):
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

            ids_a, mask_a = self._pad_id_seq(ids_a, mask_a)
            ids_b, mask_b = self._pad_id_seq(ids_b, mask_b)

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


def main():
    model_config = SentConfig.from_json_file(model_config_file)

    if max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the model "
            "was only trained up to sequence length %d" %
            (max_seq_length, model_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    processor = DataProcessor(sent_length=sent_length, tokenizer=tokenizer,
                              true_file=true_file, false_file=false_file,
                              infer_file=infer_file)

    if train:
        num_data = processor.get_num_train_data()
        num_train_steps = num_data // batch_size

        with tf.Graph().as_default() as global_graph:
            input_ids = tf.placeholder(shape=(batch_size, max_seq_length), dtype=tf.int32, name='input_ids')
            input_mask = tf.placeholder(shape=(batch_size, 2, sent_length), dtype=tf.int32, name='input_mask')
            input_sents_length = tf.placeholder(shape=(batch_size, 2), dtype=tf.int32, name='segment_ids')
            label_id = tf.placeholder(shape=(batch_size, 1), dtype=tf.int32, name='label_id')

            loss, predict = create_model(
                model_config=model_config,
                is_training=True,
                input_ids=input_ids,
                sents_length=input_sents_length,
                input_mask=input_mask,
                labels=label_id)

            num_warmup_steps = int(num_warmup_proportion * num_train_steps)
            train_op, global_step = optimization.create_optimizer(loss, learning_rate, num_train_steps,
                                                                  num_warmup_steps, use_tpu=False)

            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(graph=global_graph, config=config) as sess:
                tf.global_variables_initializer().run()

                tvars = tf.trainable_variables() + [global_step]
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

                saver = tf.train.Saver(tf.trainable_variables() + [global_step], max_to_keep=5)
                tf.summary.scalar('loss', loss)
                summary_op = tf.summary.merge_all()
                summary_writter = tf.summary.FileWriter(os.path.join(output_dir, 'train_summary'), global_graph)

                for e in range(num_epoch):
                    tq = tqdm(range(num_train_steps))
                    for step in tq:
                        ids, masks, sents_length, labels = processor.get_train_data(batch_size)

                        sess.run(train_op,
                                 feed_dict={
                                     input_ids: ids,
                                     input_mask: masks,
                                     input_sents_length: sents_length,
                                     label_id: labels
                                 })

                        if step % 100 == 0:
                            summary = sess.run(summary_op,
                                               feed_dict={
                                                   input_ids: ids,
                                                   input_mask: masks,
                                                   input_sents_length: sents_length,
                                                   label_id: labels
                                               })
                            summary_writter.add_summary(summary, global_step.eval(session=sess))

                            if step % 1000 == 0:
                                saver.save(sess, output_dir + '/ckpt', global_step=global_step)

    elif infer:

        with tf.Graph().as_default() as global_graph:
            input_ids = tf.placeholder(shape=(1, max_seq_length), dtype=tf.int32, name='input_ids')
            input_mask = tf.placeholder(shape=(1, 2, sent_length), dtype=tf.int32, name='input_mask')
            input_sents_length = tf.placeholder(shape=(1, 2), dtype=tf.int32, name='segment_ids')
            label_id = tf.placeholder(shape=(1, 1), dtype=tf.int32, name='label_id')

            loss, predict = create_model(
                model_config=model_config,
                is_training=True,
                input_ids=input_ids,
                sents_length=input_sents_length,
                input_mask=input_mask,
                labels=label_id)

            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(graph=global_graph, config=config) as sess:
                tf.global_variables_initializer().run()

                tvars = tf.trainable_variables()
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

                infer_data = processor.prepare_infer_data()
                length = len(infer_data)
                match_found = False

                for sent1_idx in tqdm(range(infer_start_index, length)):
                    for sent2_idx in tqdm(sample(range(length), min(length, num_sent_to_compare))):
                        if match_found:
                            match_found = False
                            break

                        if sent1_idx == sent2_idx or infer_data[sent1_idx] == infer_data[sent2_idx]:
                            continue

                        if len(infer_data[sent2_idx] < 3) or len(infer_data[sent2_idx]) > 20:
                            continue

                        if len(infer_data[sent2_idx]) > 5 * len(infer_data[sent1_idx]):
                            continue

                        ids, masks, sents_length, labels = \
                            processor.create_infer_data(infer_data[sent1_idx], infer_data[sent2_idx])

                        pred = sess.run(predict,
                                        feed_dict={
                                            input_ids: ids,
                                            input_mask: masks,
                                            input_sents_length: sents_length,
                                            label_id: labels

                                        })

                        if infer_lower_bound < pred < infer_upper_bound:
                            output_file = os.path.join(infer_output_dir, 'predictions.tsv')
                            with tf.gfile.GFile(output_file, "a") as writer:
                                result_line = ' && '.join(
                                    [infer_data[sent1_idx], infer_data[sent2_idx], str(pred)]
                                ) + '\n'

                                writer.write(result_line)

                            match_found = True


if __name__ == "__main__":
    main()
