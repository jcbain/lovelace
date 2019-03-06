import datetime
import pickle

import tensorflow as tf

from random import shuffle
from numpy import copy
from nn.scully.gatedrnn import GatedRNN
from workers.dataloaders.data_fiefdom import DataOverlord
from workers.dataloaders.duster import *

tf.flags.DEFINE_string('data_file', 'data/sample/data.csv',
                       """Data file containing text along with their labels (must have the columns `SentimentText` and 
                       `Sentiment`).""")
tf.flags.DEFINE_string('embedding_file', 'pretrained_embeddings/sample_embeddings.npz',
                       """.npz file containing the vocab along with the pre-trained word embeddings (must have the 
                       objects `embeddings` and `vocab`).""")
tf.flags.DEFINE_integer('sequence_len', None,
                        'The length of the tensors containing word indexes.')
tf.flags.DEFINE_float('test_size', 0.2,
                      'Proportion of `data_file` that should be withheld as testing set.')
tf.flags.DEFINE_integer('valid_size', 100,
                        'Number of rows to independently validate with during training.')
tf.flags.DEFINE_boolean('remove_rt', True,
                        'An option to remove the retweets from `data_file`.')
tf.flags.DEFINE_boolean('remove_mentioner', True,
                        'An option to remove the mentioner (@twitter_handle) from `data_file`.')
tf.flags.DEFINE_integer('batch_size', 40,
                        'The number of rows per batch.')
tf.flags.DEFINE_integer('hidden_dim', 64,
                        'Number of units in the hidden dimension.')
tf.flags.DEFINE_integer('num_classes', 2,
                        'Number of distinct classes in the target variable.')
tf.flags.DEFINE_integer('num_epochs', 1000,
                        'Number of epochs to run the network.')
tf.flags.DEFINE_integer('embed_dim', 50,
                        'The length of each embedded word.')
tf.flags.DEFINE_integer('num_layers', 1,
                        'The number of gated cells in the network.')
tf.flags.DEFINE_string('cell_type', 'lstm',
                       'The type of cell between the options of "LSTM" and "GRU".')
tf.flags.DEFINE_float('learning_rate', 0.001,
                      'The rate at which to set the optimizer to learn.')
tf.flags.DEFINE_float('dropout_keep', 0.5,
                      '0<dropout_keep<=1. Dropout keep-probability.')
tf.flags.DEFINE_string('checkpoints_dir', 'checkpoints',
                       'The directory where the parameters will be saved.')
tf.flags.DEFINE_string('summaries_dir', 'logs',
                       'The directory where the summaries will be stored.')
tf.flags.DEFINE_integer('write_summary_every', 2,
                        'How often to write a summary.')
tf.flags.DEFINE_integer('validate_every', 10,
                        'How often to validate the model.')
tf.flags.DEFINE_integer('add_checkpoint', 50,
                        'How often to overwrite the checkpoint.')
tf.flags.DEFINE_boolean('opportunistic_save', True,
                        'Option to save checkpoint when loss is less than previous test and accuracy is higher.')

FLAGS = tf.flags.FLAGS

# Prepare summaries
summaries_dir = '{0}/{1}/{2}'.format(FLAGS.summaries_dir,
                                     'rnn',
                                     datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))

train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Prepare model directory
#model_name = str(int(time.time()))
#model_dir = '{0}/{1}/{2}'.format(summaries_dir,FLAGS.checkpoints_dir, model_name)
#if not os.path.exists(model_dir):
#    os.makedirs(model_dir)

# Save configuration
if tf.__version__ in ['1.2.1', '1.4.0']:
    FLAGS._parse_flags()
    config = FLAGS.__dict__['__flags']
elif tf.__version__ in ['1.5.0', '1.5.1', '1.12.0']:
    config =  FLAGS.flag_values_dict()
with open('{}/config.pkl'.format(summaries_dir), 'wb') as f:
    pickle.dump(config, f)

# Prep data
do = DataOverlord(data_file=FLAGS.data_file, embedding_file=FLAGS.embedding_file, sequence_len=FLAGS.sequence_len,
                  test_size=FLAGS.test_size, val_samples=FLAGS.valid_size, remove_retweets=FLAGS.remove_rt,
                  remove_mentioner=FLAGS.remove_mentioner)

# Save vocab and sequence length
vocab_atts = {
    'sequence_len': len(do.tensors[0]),
    'vocab': do.vocab
}
with open('{}/vocab_atts.pkl'.format(summaries_dir), 'wb') as f:
    pickle.dump(vocab_atts, f)

# Save data attribtutes
data_atts = {
    'testing_indexes': do.test_indexes,
    'validate_indexes': do.val_indexes,
    'train_indexes': do.train_indexes
}
with open('{}/data_atts.pkl'.format(summaries_dir), 'wb') as f:
    pickle.dump(data_atts, f)

# Build computational graph
nn = GatedRNN(hidden_dim=FLAGS.hidden_dim, num_classes=FLAGS.num_classes,
              vocab_size=do.vocab_size, seq_len=do.tensors.shape[1], embed_dim=FLAGS.embed_dim, 
              num_layers=FLAGS.num_layers, cell_type=FLAGS.cell_type, learning_rate=FLAGS.learning_rate,
              word_vectors=do.embeddings)

# Set up vocab metadata
# metadata = "{}/metadata.tsv".format(summaries_dir)

# with open(metadata, 'w') as metadata_file:
#     for row in do.vocab:
#         metadata_file.write('{}\n'.format(row))

# Train model
sess = tf.Session()
sess.run(nn.initialize_all_variables())
saver = tf.train.Saver()
x_val, y_val = do.tensors[do.val_indexes], do.sentiments[do.val_indexes]
data, targets = do.tensors, do.sentiments
batch_indexes = make_batch_indexes(FLAGS.batch_size, len(do.train_indexes))
previous_accuracy = 0

for i in range(FLAGS.num_epochs):
    epoch_indexes = copy(do.train_indexes)
    shuffle(epoch_indexes)
    batch_generator = create_batch(batch_indexes)
    try:
        if (i + 1) % FLAGS.write_summary_every == 0:
            accuracy, loss, summary, embeds = sess.run([nn.accuracy, nn.loss, nn.merged, nn.embeddings],
                                                       feed_dict={nn.inputs: batch_x,
                                                                  nn.target: batch_y,
                                                                  nn.dropout: FLAGS.dropout_keep})
            train_writer.add_summary(summary, i)
            print("epoch {0}: loss= {1:.4f} | accuracy= {2:.4f}".format(i, loss, accuracy))
    except:
        pass

    try:
        if (i + 1) % FLAGS.validate_every == 0:
            val_loss, val_accuracy, val_summary = sess.run([nn.loss, nn.accuracy, nn.merged],
                                                           feed_dict={nn.inputs: x_val,
                                                                      nn.target: y_val,
                                                                      nn.dropout: 1.0})

            validation_writer.add_summary(val_summary, i)
            print("  VALIDATION LOSS: {0:.4f} (accuracy {1:.4f})".format(val_loss, val_accuracy))

            if FLAGS.opportunistic_save:
                if previous_accuracy <= val_accuracy:
                    checkpoint_file = '{}/model.ckpt'.format(summaries_dir)
                    save_path = saver.save(sess, checkpoint_file)
                    print('Model saved in: {0}'.format(summaries_dir))
                    previous_accuracy = val_accuracy

    except:
        pass

    try:
        if (i + 1) % FLAGS.add_checkpoint == 0:
            if not FLAGS.opportunistic_save:
                checkpoint_file = '{}/model.ckpt'.format(summaries_dir)
                save_path = saver.save(sess, checkpoint_file)
                print('Model saved in: {0}'.format(summaries_dir))
    except:
        pass

    for j in range(len(batch_indexes)):
        start_end_ind = next(batch_generator)
        batch_x = data[epoch_indexes[start_end_ind[0]: start_end_ind[1]]]
        batch_y = targets[epoch_indexes[start_end_ind[0]: start_end_ind[1]]]
        sess.run(nn.optimizer, feed_dict={nn.inputs: batch_x,
                                          nn.target: batch_y,
                                          nn.dropout: FLAGS.dropout_keep})

        # Write summary to tensorboard
        #if (j + 1) % FLAGS.write_summary_every == 0:
        #   accuracy, loss, summary, embeds = sess.run([nn.accuracy, nn.loss, nn.merged, nn.embeddings],
        #                                               feed_dict={nn.inputs: batch_x,
        #                                                          nn.target: batch_y,
        #                                                          nn.dropout: FLAGS.dropout_keep})

        #    train_writer.add_summary(summary, i)
        #    print("epoch {0}: loss= {1:.4f} | accuracy= {2:.4f}".format(i, loss, accuracy))



        # Check validation performance
        #if (j + 1) % FLAGS.validate_every == 0:
        #    val_loss, val_accuracy, val_summary = sess.run([nn.loss, nn.accuracy, nn.merged],
        #                                                   feed_dict={nn.inputs: x_val,
        #                                                              nn.target: y_val,
        #                                                              nn.dropout: 1.0})
        #    validation_writer.add_summary(val_summary, i)
        #    print("  VALIDATION LOSS: {0:.4f} (accuracy {1:.4f})".format(val_loss, val_accuracy))


#     if (i + 1) == FLAGS.num_epochs:
#         embeddings = sess.run(nn.embeddings, feed_dict={nn.inputs: batch_x,
#                                                        nn.target: batch_y})

#         embedding_var = embeddings 
#         sess.run(embedding_var.initializer)
#         print(embedding_var)
#         # adding into projector
#         proj_config = projector.ProjectorConfig()
#         embed = proj_config.embeddings.add()
#         embed.tensor_name = embedding_var.name
#         embed.metadata_path = metadata 

#         # Specify the width and height of a single thumbnail.
#         projector.visualize_embeddings(writer, proj_config)


checkpoint_file = '{}/model.ckpt'.format(summaries_dir)
save_path = saver.save(sess, checkpoint_file)
print('Model saved in: {0}'.format(summaries_dir))
