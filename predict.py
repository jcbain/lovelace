import csv
import pickle
import numpy as np
import tensorflow as tf
from workers.dataloaders.data_fiefdom import DataPleb

tf.flags.DEFINE_string('logdir', 'logs/cnn/22_Feb_2018-20_22_02',
                       'Logs directory (example: logs/cnn/22_Feb_2018-20_22_02 ). Must contain (at least):\n'
                       '- config.pkl: Contains the model training parameters \n'
                       '- model.ckpt: Contains the model weights \n'
                       '- model.ckpt.meta: Contains the TensorFlow graph definition \n'
                       '- vocab_atts.pkl: Contains the vocab and sequence length of the training data \n')
tf.flags.DEFINE_string('data_file', None,
                       'File with containing texts to be predicted. Must contain the a `SentimentText` column.')
tf.flags.DEFINE_string('single_text', None,
                       'A single bit of text to be predicted.')
tf.flags.DEFINE_string('output_file', None,
                       'File to be written.')
tf.flags.DEFINE_bool('p', False,
                     'Print output to terminal.')
tf.flags.DEFINE_string('text_name', 'text',
                       'Name of the attribute containing the text to be predicted off of.')
tf.flags.DEFINE_string('extra_atts', 'tweet_id_str,job_id,created_at',
                       'Extra columns to be appended to the results.')
tf.flags.DEFINE_string('encoding', None,
                       'Ecoding of the file.')
tf.flags.DEFINE_string('delimiter', '|',
                       'Delimiter for the file.')
tf.flags.DEFINE_string('qchar', '&',
                       'Quoted character for the file.')



FLAGS = tf.flags.FLAGS

# Load configuration
with open('{}/config.pkl'.format(FLAGS.logdir), 'rb') as f:
    config = pickle.load(f)

# Load vocab
with open('{}/vocab_atts.pkl'.format(FLAGS.logdir), 'rb') as f:
    vocab_atts = pickle.load(f)

# split extra_atts option into a list if it exists
if FLAGS.extra_atts is not None:
    extra_atts = [str(i) for i in FLAGS.extra_atts.split(',')]
else:
    extra_atts = None

# Turn on print option if single_text
if FLAGS.single_text is not None:
    FLAGS.p = True

dp = DataPleb(data_file=FLAGS.data_file,
              indiv_text=FLAGS.single_text,
              vocab=vocab_atts['vocab'],
              sequence_len=vocab_atts['sequence_len'],
              text_name=FLAGS.text_name,
              extra_atts=extra_atts,
              encoding=FLAGS.encoding,
              sep=FLAGS.delimiter,
              qchar=FLAGS.qchar)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    # Import graph and restore its weights
    print('Restoring the graph ...')
    saver = tf.train.import_meta_graph("{}/model.ckpt.meta".format(FLAGS.logdir))
    saver.restore(sess, ("{}/model.ckpt".format(FLAGS.logdir)))

    # Recover input/output tensors
    input = graph.get_operation_by_name('input').outputs[0]
    target = graph.get_operation_by_name('target').outputs[0]
    dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
    predict = graph.get_operation_by_name('final_layer/softmax/predictions').outputs[0]
    accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

    # Perform prediction
    pred = sess.run([predict],
                    feed_dict={input: dp.tensors,
                               dropout_keep_prob: 1})

# print results
outputs = []
for i in list(pred):
    for j in i:
        outputs.append(np.array(j))
        if FLAGS.p:
            print("classification : {} ({})".format(np.argmax(j), j[np.argmax(j)]))

outputs = np.array(outputs)

if FLAGS.single_text is None:
    meta_out = np.concatenate((dp.atts, outputs), axis=1)
    if FLAGS.output_file is not None:
        with open(FLAGS.output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(meta_out)
            print("file written to {}".format(FLAGS.output_file))

