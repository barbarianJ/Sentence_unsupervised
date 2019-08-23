# encoding=utf-8
import sys
sys.path.append('../../gdu')
import codecs
import gdu as g
from GRU_model.sents_model import Cls
import tf_common as tc
from random import shuffle, sample
from utils import jieba_tokenization as tokenization
from tqdm import tqdm

tf = tc.tf

PARAM = {'lr': 0.0005,
         'lr_drate': 0.9995,
         'lr_dstep': 1000,
         'lr_dlimit': 0.0001,
         'l2_rate': 1e-5,
         'emb_size': 344,
         'source_vocab_size': 21128,
         'gru_layers': 3,
         'cls_num': 2}

MODEL_DIR = './models'
VOCAB = './data/vocab.txt'
DATA = './data/handwritten/sents.txt'
TRAIN_STEP = 1000000
BATCH_SIZE = 27 * 2
SAVING_STEP = 10000

# ***************************************
predict_index = 2105
predict_file = './data/crawled/crawled.txt'
predict_threshold = 0.998
predict_pair_number = 100000
predict_batch_size = 20000


class Statistics(object):
    def __init__(self):
        self.prob_sum = 0.0
        self.std = 0.0
        self.largest = 0.0
        self.counter = 0
        self.sent2_id = None

    def set_sent2_id(self, sent2_id):
        self.sent2_id = sent2_id


def read_content(f):
    content = []
    with codecs.open(f, 'r', encoding='utf-8') as f:
        for line in f:
            sents, label = line.split('=')
            sent1, sent2 = sents.split(' && ')

            content.append((sent1, sent2, label))
    return content


def read_predict_content(f):
    content = []
    with codecs.open(f, 'r', encoding='utf-8') as f:
        for line in f:
            content.extend(line.strip().split(' && '))

    return content


def sent_to_id(sent, tokenizer):
    token = tokenizer.tokenize(sent)
    ids = tokenizer.convert_tokens_to_ids(token)

    return ids


def create_predict_data(sent1, sent2, tokenizer):
    id1 = sent_to_id(sent1, tokenizer)
    id2 = sent_to_id(sent2, tokenizer)

    return [id1], [id2]


def create_predict_batch(sents, idx1, idx2s, tokenizer):
    input_id1 = [sent_to_id(sents[idx1], tokenizer)] * len(idx2s)
    input_id2 = []
    input_len2 = []
    max_len = 0

    for idx2 in idx2s:
        id2 = sent_to_id(sents[idx2], tokenizer)
        input_id2.append(id2)

        len2 = len(id2)
        input_len2.append(len2)
        max_len = max(max_len, len2)

    input_id2 = [ids + [3] * (max_len - len(ids)) for ids in input_id2]

    return input_id1, input_id2, input_len2


def compute_predict_static(true_prob, true_statistic,
                           false_prob, false_statistic,
                           idx2s):
    """
    true_prob, false_prob: 1D np array, ranges from 0.0 to 1.0,
    NOT filter to be larger than 0.5
    """
    true_statistic.counter += len(true_prob[true_prob >= 0.5])
    false_statistic.counter += len(false_prob[false_prob > 0.5])

    true_statistic.prob_sum += true_prob[true_prob >= 0.5].sum()
    false_statistic.prob_sum += false_prob[false_prob > 0.5].sum()

    true_arg_max = true_prob.argmax()
    false_arg_max = false_prob.argmax()

    if true_prob[true_arg_max] > true_statistic.largest:
        true_statistic.largest = true_prob[true_arg_max]
        true_statistic.sent2_id(idx2s[true_arg_max])

    if false_prob[false_arg_max] > false_statistic.largest:
        false_statistic.largest = false_prob[false_arg_max]


def compute_running_std():
    pass


def create_train_batch(content, batch_size, index, tokenizer):
    if index + batch_size > len(content):
        shuffle(content)
        index = 0

    input_id1 = []
    input_id2 = []
    input_len1 = []
    input_len2 = []
    input_label = []
    max_len = 0

    # nlp to ids
    for sent1, sent2, label in content[index: index + batch_size]:
        # token1 = tokenizer.tokenize(sent1)
        # token2 = tokenizer.tokenize(sent2)
        #
        # id1 = tokenizer.convert_tokens_to_ids(token1)
        # id2 = tokenizer.convert_tokens_to_ids(token2)

        id1 = sent_to_id(sent1, tokenizer)
        id2 = sent_to_id(sent2, tokenizer)

        len1 = len(id1)
        len2 = len(id2)
        max_len = max(max_len, max(len1, len2))

        input_id1.append(id1)
        input_id2.append(id2)
        input_len1.append(len1)
        input_len2.append(len2)
        input_label.append(int(label))

    # padding symbol: 3
    input_id1 = [ids + [3] * (max_len - len(ids)) for ids in input_id1]
    input_id2 = [ids + [3] * (max_len - len(ids)) for ids in input_id2]

    return input_id1, input_id2, input_len1, input_len2, input_label


def train():
    # begin graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as global_graph:
        model = Cls(param=PARAM, mode='train')
        # make tf config
        sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        sess_conf.gpu_options.allow_growth = True
        with tf.Session(graph=global_graph, config=sess_conf) as sess:
            model, global_step = tc.load_model(model, MODEL_DIR, sess)
            sess.graph.finalize()
            g.rainbow('          |BEGIN GLOBAL STEP : %d|          ' % global_step)

            # [(sent1, sent2, label)]
            content = read_content(DATA)
            shuffle(content)
            tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB)

            index = 0
            for i in range(TRAIN_STEP):

                feed_val = create_train_batch(content=content, batch_size=BATCH_SIZE, index=index, tokenizer=tokenizer)
                index += BATCH_SIZE

                _, ce_loss, l2_loss, global_step, lr, gn = model.train(sess, feed_val)
                g.rainbow(
                    'CE_LOSS:%.4f, L2LOSS:%.4f STEP:%d (LR:%.6f) gn:%3f            '
                    % (ce_loss, l2_loss, global_step, lr, gn))
                g.record('CELOSS', ce_loss)
                g.record('L2LOSS', l2_loss)
                g.record('TLOSS', (ce_loss + l2_loss) / 2.0)
                # save model
                if global_step > 0 and global_step % SAVING_STEP == 0:
                    model.saver.save(sess, MODEL_DIR + '/cls.ckpt', global_step=global_step)
                    g.warn('                                                                         ')
                    g.warn('                      SAVE MODEL WHEN STEP TO :%d                      ' % global_step)
                    g.warn('                                                                         ')


def predict():
    with tf.Graph().as_default() as global_graph:
        model = Cls(param=PARAM, mode='predict')

        sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        sess_conf.gpu_options.allow_growth = True
        with tf.Session(graph=global_graph, config=sess_conf) as sess:
            model, global_step = tc.load_model(model, MODEL_DIR, sess)

            sents = read_predict_content(predict_file)
            length = len(sents)
            tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB)

            for idx1 in tqdm(range(predict_index, length)):

                num_to_sample = predict_pair_number // predict_batch_size

                true_statistic = Statistics()
                false_statistic = Statistics()

                for _ in tqdm(range(num_to_sample)):
                    idx2s = sample(range(length), predict_batch_size)

                    feed_val = create_predict_batch(sents, idx1, idx2s, tokenizer)

                    # [B, 2]
                    _, probability = model.predict(sess, feed_val)

                    true_prob = probability[:, 0]
                    false_prob = probability[:, 1]

                    compute_predict_static(true_prob, true_statistic,
                                           false_prob, false_statistic,
                                           idx2s)

                print_str = sents[idx1] + ' && '
                if true_statistic.sent2_id is not None:
                    print_str += sents[true_statistic.sent2_id]
                print_str = '%-40s' % print_str
                print_statistic = 'num_true: %d, num_false: %d, max_prob: %.4f, true_mean_prob: %.4f, false_mean_prob: %.4f' \
                                  % (true_statistic.counter, false_statistic.counter, true_statistic.largest,
                                     true_statistic.prob_sum / float(true_statistic.counter),
                                     false_statistic.prob_sum / float(false_statistic.counter))

                if true_statistic.largest > predict_threshold:
                    g.writef('predict_true.txt', print_str + print_statistic)
                    g.writef('prediction.txt', print_str.strip())
                else:
                    g.writef('predict_false.txt', print_str + print_statistic)


if __name__ == '__main__':
    argv = sys.argv
    mode = 'train'
    if len(argv) == 2:
        mode = str(argv[1])
    g.rainbow('MODE: ' + mode)
    if mode == 'train':
        train()
    else:
        predict()
