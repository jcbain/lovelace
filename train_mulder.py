import datetime
import pickle

import tensorflow as tf

from nn.mulder.cnn import CNN
from workers.dataloaders.data_fiefdom import DataOverlord

tf.flags.DEFINE_string('data_file', 'data/sample/data.csv',
                       """Data file containing text along with their labels (must have the columns `Text` and 
                       `Sentiment`).""")
tf.flags.DEFINE_string('embedding_file', 'pretrained_embeddings/sample_embeddings.npz',
                       """.npz file containing the vocab along with the pre-trained word embeddings (must have the 
                       objects `embeddings` and `vocab`).""")
tf.flags.DEFINE_integer('sequence_len', None,
                        'The length of the tensors containing word indexes.')
tf.flags.DEFINE_float('test_size', 0.2,
                      'Proportion of `data_file` that should be withheld as testing set.')
tf.flags.DEFINE_boolean('remove_rt', True,
                        'An option to remove the retweets from `data_file`.')
tf.flags.DEFINE_boolean('remove_mentioner', True,
                        'An option to remove the mentioner (@twitter_handle) from `data_file`.')
tf.flags.DEFINE_integer('batch_size', 40,
                        'The number of rows per batch.')
tf.flags.DEFINE_integer('num_classes', 2,
                        'Number of distinct classes in the target variable.')
tf.flags.DEFINE_integer('num_epochs', 1000,
                        'Number of epochs to run the network.')
tf.flags.DEFINE_integer('embed_dim', 50,
                        'The length of each embedded word.')
tf.flags.DEFINE_integer('num_layers', 2,
                        'The number of fully connected convolution layers in the network.')
tf.flags.DEFINE_string('filter_sizes', '3,4,5',
                       'Size of the convolution filer windows.')
tf.flags.DEFINE_integer('num_filters', 128,
                        'Number of filters.')
tf.flags.DEFINE_float('l2_regulizer_lambda', 0.0,
                      'L2 regulizer lambda value.')
tf.flags.DEFINE_string('activation_function', 'tanh',
                       'Activation function for connected layers.')
tf.flags.DEFINE_float('learning_rate', 0.001,
                      'The rate at which to set the optimizer to learn.')
tf.flags.DEFINE_float('dropout_keep', 0.5,
                      '0<dropout_keep<=1. Dropout keep-probability.')
tf.flags.DEFINE_string('checkpoints_dir', 'checkpoints',
                       'The directory where the parameters will be saved.')
tf.flags.DEFINE_string('summaries_dir', 'logs',
                       'The directory where the summaries will be stored.')
tf.flags.DEFINE_integer('write_summary_every',50,
                        'How often to write a summary.')
tf.flags.DEFINE_integer('validate_every', 100,
                        'How often to validate the model.')

FLAGS = tf.flags.FLAGS

# Prepare summaries
summaries_dir = '{0}/{1}/{2}'.format(FLAGS.summaries_dir,
                                     'cnn',
                                     datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))

train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Save configuration
if tf.__version__ in ['1.2.1', '1.4.0']:
    FLAGS._parse_flags()
    config = FLAGS.__dict__['__flags']
elif tf.__version__ in ['1.5.0', '1.5.1']:
    config = FLAGS.flag_values_dict()
with open('{}/config.pkl'.format(summaries_dir), 'wb') as f:
    pickle.dump(config, f)

# Prep data
do = DataOverlord(data_file=FLAGS.data_file, embedding_file=FLAGS.embedding_file, sequence_len=FLAGS.sequence_len,
                  test_size=FLAGS.test_size, val_samples=FLAGS.batch_size, remove_retweets=FLAGS.remove_rt,
                  remove_mentioner=FLAGS.remove_mentioner)

# Save vocab and sequence length
vocab_atts = {
    'sequence_len': len(do.tensors[0]),
    'vocab': do.vocab
}
with open('{}/vocab_atts.pkl'.format(summaries_dir), 'wb') as f:
    pickle.dump(vocab_atts, f)


# Make FLAGS.fiter_sizes into a list
filter_sizes = [int(i) for i in FLAGS.filter_sizes.split(',')]

nn = CNN(num_classes=FLAGS.num_classes, vocab_size=do.vocab_size,
         seq_len=do.tensors.shape[1], embed_dim=FLAGS.embed_dim, filter_sizes=filter_sizes,
         num_filters=FLAGS.num_filters, learning_rate=FLAGS.learning_rate, word_vectors=do.embeddings,
         l2_reg_lambda=FLAGS.l2_regulizer_lambda, fc_layers=FLAGS.num_layers, 
         activation_func=FLAGS.activation_function)

sess = tf.Session()
sess.run(nn.initialize_all_variables())
saver = tf.train.Saver()
x_val, y_val, val_seq_len = do.get_val_data()

for i in range(FLAGS.num_epochs):
    batch_x, batch_y, _ = do.next_batch(FLAGS.batch_size)
    sess.run(nn.optimizer, feed_dict={nn.inputs: batch_x,
                                      nn.target: batch_y,
                                      nn.dropout: FLAGS.dropout_keep})
    
    # Write summary to tensorboard
    if (i + 1) % FLAGS.write_summary_every == 0:
        accuracy, loss, summary, embeds = sess.run([nn.accuracy, nn.loss, nn.merged, nn.embeddings],
                                                   feed_dict={nn.inputs: batch_x,
                                                              nn.target: batch_y,
                                                              nn.dropout: FLAGS.dropout_keep})
        train_writer.add_summary(summary, i)
        print("epoch {0}: loss= {1:.4f} | accuracy= {2:.4f}".format(i, loss, accuracy))

    # Check validation performance
    if (i + 1) % FLAGS.validate_every == 0:
        val_loss, val_accuracy, val_summary = sess.run([nn.loss, nn.accuracy, nn.merged],
                                                       feed_dict={nn.inputs: x_val,
                                                                  nn.target: y_val,
                                                                  nn.dropout: 1.0})
        validation_writer.add_summary(val_summary, i)
        print("  VALIDATION LOSS: {0:.4f} (accuracy {1:.4f})".format(val_loss, val_accuracy))

checkpoint_file = '{}/model.ckpt'.format(summaries_dir)
save_path = saver.save(sess, checkpoint_file)
print('Model saved in: {0}'.format(summaries_dir))