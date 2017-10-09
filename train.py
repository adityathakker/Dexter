import sugartensor as tf
from data import SpeechCorpus, vocab_size
from model import *

tf.sg_verbosity(10)

batch_size = 16   

data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
labels = tf.split(data.label, tf.sg_gpus(), axis=0)

seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))


@tf.sg_parallel
def get_loss(opt):
    logit = get_logit(opt.input[opt.gpu_index], voca_size=vocab_size)
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])

tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
            ep_size=data.num_batch, max_ep=50)
