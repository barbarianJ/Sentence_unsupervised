# encoding=utf-8
import tensorflow as tf
import sys
sys.path.append('../../gdu')
import gdu as g
import tf_common as tc

scope = tf.variable_scope


class Cls(object):

    def __init__(self, param, mode='train'):
        g.rainbow('INIT MODEL Class...')
        self.lr = param['lr']
        self.lr_drate = param['lr_drate']
        self.lr_dstep = param['lr_dstep']
        self.lr_dlimit = param['lr_dlimit']
        self.l2_rate = param['l2_rate']
        self.emb_size = param['emb_size']
        self.source_vs = param['source_vocab_size']
        self.gru_layers = param['gru_layers']
        self.cls_num = param['cls_num']
        self.mode = mode

        self.input_data1 = tf.placeholder(tf.int32, [None, None], 'input_data1')
        self.input_data2 = tf.placeholder(tf.int32, [None, None], 'input_data2')
        self.input_data_len1 = tf.placeholder(tf.int32, [None], 'input_data_len1')
        self.input_data_len2 = tf.placeholder(tf.int32, [None], 'input_data_len2')
        self.label = tf.placeholder(tf.int32, [None], 'label')
        self.batch_size = tf.size(self.input_data_len1)

        if mode == 'predict':
            self.input_data_len1 = None

        self.build()
        self.loss()
        self.optim()

    def set_global_init(self, init):
        tf.get_variable_scope().set_initializer(init)

    def build(self):
        # create global step
        self.global_step = tf.Variable(0, trainable=False)
        # create embedding table
        self.enc_emb_table = tc.random_embeddings(self.source_vs, self.emb_size, scope_name='encoder')
        # get specific embs
        self.enc_emb_input1 = tc.emb_lookup(self.enc_emb_table, self.input_data1)
        self.enc_emb_input2 = tc.emb_lookup(self.enc_emb_table, self.input_data2)
        self.enc_emb_input1 = tc.l2n(self.enc_emb_input1)
        self.enc_emb_input2 = tc.l2n(self.enc_emb_input2)
        # do self attention for some iter

        with scope('encoder1'):
            fwl1, bwl1 = tc.bi_gru(self.emb_size, self.gru_layers, ac=tc.ngelu, dropout=0.1, mode=self.mode,
                                   resc=(self.gru_layers - 1))
            # bi_state: (fw, bw)
            # fw_state: (c: state, h: output)
            bi_out1, bi_state1 = tf.nn.bidirectional_dynamic_rnn(fwl1, bwl1, self.enc_emb_input1, dtype=tf.float32,
                                                                 sequence_length=self.input_data_len1)
            self.state1 = bi_state1[0][0] + bi_state1[1][0]

        with scope('encoder2'):
            fwl2, bwl2 = tc.bi_gru(self.emb_size, self.gru_layers, ac=tc.ngelu, dropout=0.1, mode=self.mode,
                                   resc=(self.gru_layers - 1))
            bi_out2, bi_state2 = tf.nn.bidirectional_dynamic_rnn(fwl2, bwl2, self.enc_emb_input2, dtype=tf.float32,
                                                                 sequence_length=self.input_data_len2)
            self.state2 = bi_state2[0][0] + bi_state2[1][0]

        with scope('attention'):
            # [B, E]
            concate_state = tf.concat((self.state1, self.state2), axis=-1)

            attention = tc.selfatt(concate_state, self.emb_size * 2, mode=self.mode)
            attention = tc.ngelu(attention)
            attention = tc.selfatt(attention, self.emb_size * 2, mode=self.mode)
            attention = tc.ngelu(attention)

        with scope('projection'):
            self.logits = tc.dense(self.cls_num, ac=None)(attention)
            g.rainbow('logits')
            g.rainbow(self.logits)
            self.finalr = tf.argmax(self.logits, axis=-1)

    def loss(self):
        self.l2_loss = tc.l2_reg(self.l2_rate) / tf.cast(self.batch_size, tf.float32)
        ohlabel = tf.one_hot(self.label, self.cls_num)
        self.ce_loss = tf.reduce_mean(tc.ce(labels=ohlabel, logits=self.logits))
        self.loss = self.l2_loss + self.ce_loss

    def optim(self):
        self.lr = tf.maximum(tf.constant(self.lr_dlimit),
                             tf.train.exponential_decay(self.lr, self.global_step, self.lr_dstep, self.lr_drate,
                                                        staircase=True))
        self.opt = tf.train.AdamOptimizer(self.lr)
        params = tf.trainable_variables()
        grds = tf.gradients(self.loss, params)
        grds, self.gn = tc.gclip(grds)
        self.update = self.opt.apply_gradients(zip(grds, params), global_step=self.global_step)
        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def train(self, session, value):
        return session.run([self.update, self.ce_loss, self.l2_loss, self.global_step, self.lr, self.gn],
                           self._make_train_feed_dict(*value))

    def infer(self, session, value):
        return session.run([self.finalr, self.logits], self._make_infer_feed_dict(*value))

    def predict(self, session, value):
        prob = tf.nn.softmax(self.logits, axis=-1)
        return session.run([self.logits, prob], self._make_predict_feed_dict(*value))

    def _make_train_feed_dict(self, input_data1, input_data2, input_data_len1, input_data_len2, label):
        return {self.input_data1: input_data1,
                self.input_data2: input_data2,
                self.input_data_len1: input_data_len1,
                self.input_data_len2: input_data_len2,
                self.label: label}

    def _make_infer_feed_dict(self, input_data1, input_data2, input_data_len1, input_data_len2):
        return {self.input_data1: input_data1,
                self.input_data2: input_data2,
                self.input_data_len1: input_data_len1,
                self.input_data_len2: input_data_len2}

    def _make_predict_feed_dict(self, input_data1, input_data2, input_data_len2):
        return {self.input_data1: input_data1,
                self.input_data2: input_data2,
                self.input_data_len2: input_data_len2}
