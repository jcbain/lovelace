import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from workers.dataloaders.data_fiefdom import DataOverlord

tf.flags.DEFINE_string('logdir', 'logs/rnn/25_Jan_2019-11_37_58',
                       'The log directory.')
tf.flags.DEFINE_string('output', '~/Desktop/tmp.csv',
                       'The output .csv file.')

FLAGS = tf.flags.FLAGS

# read in the configuration file
with open('{}/config.pkl'.format(FLAGS.logdir), 'rb') as f:
    config = pickle.load(f)

# read in the data attributes
with open('{}/data_atts.pkl'.format(FLAGS.logdir), 'rb') as f:
    data_atts = pickle.load(f)

do = DataOverlord(data_file=config['data_file'], embedding_file=config['embedding_file'],
                  sequence_len=config['sequence_len'], test_size=config['test_size'],
                  val_samples=config['valid_size'], remove_retweets=config['remove_rt'],
                  remove_mentioner=config['remove_mentioner'])

# convert sentiment one-hots into argmax index vals
sent_vals = []
for i in do.sentiments:
    sent_vals.append(np.argmax(i))
sent_vals = np.array(sent_vals)

# create dataframe
df = pd.DataFrame({'SentimentText': do.samples[data_atts['testing_indexes']],
                   'Sentiment': sent_vals[data_atts['testing_indexes']]})

df.to_csv(FLAGS.output, index=False, sep='\t')
