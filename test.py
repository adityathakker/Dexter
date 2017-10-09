import sugartensor as tf
from data import SpeechCorpus, vocab_size
from model import *
import numpy as np
from tqdm import tqdm


tf.sg_verbosity(10)

tf.sg_arg_def(set=('valid', "'train', 'valid', or 'test'.  The default is 'valid'"))
tf.sg_arg_def(frac=(1.0, "test fraction ratio to whole data set. The default is 1.0(=whole set)"))

batch_size = 16

data = SpeechCorpus(batch_size=batch_size, set_name=tf.sg_arg().set)

x = data.mfcc
y = data.label

seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

logit = get_logit(x, voca_size=vocab_size)

loss = logit.sg_ctc(target=y, seq_len=seq_len)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    tf.sg_init(sess)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

    tf.sg_info('Testing started on %s set at global step[%08d].' %
               (tf.sg_arg().set.upper(), sess.run(tf.sg_global_step())))
    with tf.sg_queue_context():
        iterator = tqdm(range(0, int(data.num_batch * tf.sg_arg().frac)), total=int(data.num_batch * tf.sg_arg().frac),
                        initial=0, desc='test', ncols=70, unit='b', leave=False)

        loss_avg = 0.
        for _ in iterator:

            batch_loss = sess.run(loss)

            if batch_loss is not None and \
                    not np.isnan(batch_loss.all()) and not np.isinf(batch_loss.all()):
                loss_avg += np.mean(batch_loss)

        loss_avg /= data.num_batch * tf.sg_arg().frac

    tf.sg_info('Testing finished on %s.(CTC loss=%f)' % (tf.sg_arg().set.upper(), loss_avg))
