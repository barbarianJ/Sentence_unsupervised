model_config_file = 'model/model_config/config.json'
max_seq_length = 100
output_dir = 'result/'
train = True
infer = False
# train = False
# infer = True
vocab_file = 'model/model_config/vocab.txt'
do_lower_case = False
true_file = 'data/handwritten_qingyun/han_qing_true.txt'
false_file = 'data/handwritten_qingyun/han_qing_false.txt'

infer_file = 'data/crawled/crawled.txt'
infer_output_dir = 'infer/'

init_checkpoint = 'result/ckpt-23351'
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
