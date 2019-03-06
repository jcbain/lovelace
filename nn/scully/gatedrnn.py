# -*- coding: utf-8 -*-
"""Scully gated RNN neural network class.

Author: James Bain <jamescbain@mizzou.edu>
Last Edited: 2018-03-09

Builds a gated recurrent neural network for phrase/sentiment classification tasks using the tensorflow API. Can be
either a Long Short Term Memory (LSTM) or Gated Recurrent Unit (GRU) RNN. Word embeddings can either be pre-trained
or randomly initialized upon network instantiation.
"""
import tensorflow as tf


class GatedRNN(object):
    """Build a gated RNN class for sentiment classification.

    Parameters
    ----------
    hidden_dim : int
        The number of units in the hidden dimension.

    num_classes : int
        The number of distinct classes in the target.

    vocab_size : int
        The number of unique words in your training corpus.

    seq_len : int
        The number of words within each phrase. Phrases with less than the max `seq_len` are padded with 0s on the left.

    embed_dim : int
        The length of each embedded word.

    num_layers: int
        The number of gated cells in the network.

    batch_size : int
        The number of input cases per batch.

    cell_type : str
        The type of recurrent unit (either 'lstm' or 'gru').

    learning_rate : float
        The rate at which the network learns.

    word_vectors : numpy.array
        Pre-trained word vectors.
    """
    def __init__(self, hidden_dim, num_classes, vocab_size, seq_len, embed_dim, num_layers, batch_size=None,
                 cell_type="lstm", learning_rate=0.001, word_vectors=None, device='cpu', random_state=None):
        self.device = self.specify_device(device)
        self.inputs = tf.placeholder(tf.int32, [batch_size, seq_len], name="input")
        self.target = tf.placeholder(tf.float32, [batch_size, num_classes], name="target")
        self.num_classes = num_classes
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.word_vectors = word_vectors
        self.embeddings = self._embed(word_vectors=self.word_vectors, inputs=self.inputs, batch_size=batch_size,
                                      seq_len=seq_len, embed_dim=embed_dim, vocab_size=vocab_size)
        self.cell = self._create_cell(hidden_dim=hidden_dim, num_layers=num_layers, cell_type=cell_type,
                                      dropout=self.dropout)
        self.outputs = self._build_rnn(cell=self.cell, embeddings=self.embeddings)
        
        self.scores = self._scores(self.outputs, hidden_dim, num_classes)
        self.prediction = self._predict(self.scores)
        self.losses = self._losses(self.scores, self.target)
        self.accuracy = self._calculate_accuracy(self.prediction, self.target)
        self.tp, self.fn, self.fp = self._find_positives_negatives(self.prediction, self.target)
        self.recall = self._calculate_recall(self.tp, self.fn)
        self.precision = self._calculate_precision(self.tp, self.fp)
        self.loss = self._calculate_loss(self.losses)
        self.optimizer = self._train_step(self.loss, learning_rate)
        self.merged = tf.summary.merge_all()

    def _embed(self, word_vectors, inputs, batch_size, seq_len, embed_dim, vocab_size):
        """Creates a 3d tensor of word embeddings of the size [batch_size, seq_len, embed_dim].
        
        Parameters
        ----------
        inputs : tensorflow.placeholder
            A tensor of sequences with values corresponding to vocab indexes.
        
        batch_size : int
            The number of input cases per batch.
          
        seq_len : int
            The length of each sequence in the input.
        
        embed_dim : int
            The length of each word embedding.
        
        vocab_size : int
            The number of words in the entire vocab.
          
        Returns
        -------
        3d tensor of word embeddings of size [batch_size, seq_len, embed_dim].
        """
        with tf.device(self.device):
            with tf.name_scope('word_embeddings'):
                if word_vectors is None:
                    word_vect = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1, seed=1))
                else:
                    # embeddings = tf.Variable(tf.zeros([batch_size, seq_len, embed_dim]), dtype=tf.float32)
                    word_vect = tf.get_variable("embeddings", shape=[vocab_size, embed_dim],
                                                initializer=tf.constant_initializer(word_vectors))
                embeddings = tf.nn.embedding_lookup(word_vect, inputs)
        return embeddings

    def _create_cell(self, hidden_dim, num_layers, cell_type, dropout):
        """Creates a gated cell of cell_type (either "lstm" or "gru").
        
        Parameters
        ----------
        hidden_dim : int
            Length of the hidden dimension.
        
        num_layers : int
            Number of connected gated cells.
        
        cell_type : str
            Type of gated cell (either "LSTM" or "GRU").
          
        Returns
        -------
        A gated cell.
        """
        with tf.device(self.device):
            cells = []
            for _ in range(num_layers):
                if cell_type == "lstm":
                    cell = tf.contrib.rnn.LSTMCell(hidden_dim)
                elif cell_type == "gru":
                    cell = tf.contrib.rnn.GRUCell(hidden_dim)
                dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout, output_keep_prob=dropout)
                cells.append(dropout_cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)
        return cell
    
    def _build_rnn(self, cell, embeddings):
        """
        Builds the rnn layer of the network.
        
        Parameters
        ----------
        cell : tensorflow.contrib.rnn.MultiRNNCell
            The gated cell.
          
        inputs : tensorflow.placeholder
            A tensor of sequences with values corresponding to vocab indexes.
        """
        outputs, _ = tf.nn.dynamic_rnn(cell, embeddings, dtype=tf.float32)
        return outputs
    
    def _scores(self, outputs, hidden_dim, num_classes):
        """Calculates a prediction.

        Parameters
        ----------
        outputs : tensorflow.rnn.dynamic_rnn
            An rnn layer.
          
        hidden_dim : int
            Length of the hidden dimension.
          
        num_classes : int
            Number of unique classes in the target tensor.
        """
        with tf.device(self.device):
            value = tf.transpose(outputs, [1, 0, 2])

            with tf.name_scope('final_layer/weights'):
                weight = tf.Variable(tf.truncated_normal([hidden_dim, num_classes]))
                self.create_summaries(weight, 'final_layer/weights')

            with tf.name_scope('final_layer/biases'):
                bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
                self.create_summaries(bias, 'final_layer/weights')

            last = tf.gather(value, int(value.get_shape()[0]) - 1)
            scores = (tf.matmul(last, weight) + bias)
        return scores

    def _predict(self, scores):
        with tf.device(self.device):
            with tf.name_scope('final_layer/softmax'):
                softmax = tf.nn.softmax(scores, name='predictions')
                tf.summary.histogram('final_layer/softmax', softmax)
        return softmax

    def _calculate_accuracy(self, prediction, target):
        """Calculates the accuracy of the prediction.
        
        Parameters
        ----------
        prediction : tensorflow.Tensor
            A tensor of predictions.
          
        target : tensorflow.placeholder
            The actual class values.
          
        Returns
        -------
        The accuracy of the prediction. 
        """
        with tf.device(self.device):
            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
                tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _find_positives_negatives(self, prediction, target):
        """Finds the true positives, false positive and false negatives for each round of predictions.

        Parameters
        ----------
        prediction : tensorflow.Tensor
            A tensor of predictions.

        target : tensorflow.placeholder
            The actual class values.

        Returns
        -------
        tuple:
            The true positive, false positive and false negative for the round of predictions.
        """
        with tf.device(self.device):
            confusion = tf.confusion_matrix(tf.argmax(target, 1), tf.argmax(prediction, 1))
            if self.num_classes == 2:
                tp = tf.reduce_sum(tf.diag_part(confusion))
                fn = confusion[1, 0]
                fp = confusion[0, 1]
            else:
                tp = tf.diag_part(confusion)
                fn = []
                fp = []
                for i in range(0, self.num_classes):
                    fn.append(tf.reduce_sum(confusion[i, :]) - confusion[i, i])
                    fp.append(tf.reduce_sum(confusion[:, i]) - confusion[i, i])

        return tp, fn, fp

    def _calculate_recall(self, tp, fn):
        with tf.device(self.device):
            with tf.name_scope('recall'):
                recall = tf.divide(tp, tf.add(tp, fn))
                if self.num_classes == 2:
                    tf.summary.scalar('recall', recall)
                else:
                    for i in range(0, self.num_classes):
                        tf.summary.scalar('recall_{}'.format(i), recall[i])
        return recall

    def _calculate_precision(self, tp, fp):
        with tf.device(self.device):
            with tf.name_scope('precision'):
                prec = tf.divide(tp, tf.add(tp, fp))
                if self.num_classes == 2:
                    tf.summary.scalar('precision', prec)
                else:
                    for i in range(0, self.num_classes):
                        tf.summary.scalar('precision_{}'.format(i), prec[i])
        return prec

    def _losses(self, scores, target):
        with tf.device(self.device):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=target, name='cross_entropy')
        return cross_entropy
    
    def _calculate_loss(self, losses):
        with tf.device(self.device):
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(losses, name='loss')
                tf.summary.scalar('loss', loss)
        return loss
    
    def _train_step(self, loss, learning_rate):
        """Create an a function to optimize learning.
        
        Parameters
        ----------
        loss : tensorflow.scalar
            The loss of the prediction.
          
        learning_rate : float
            The learning rate
          
        Returns
        -------
        An Adam Optimizer
        """
        with tf.device(self.device):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return optimizer

    def initialize_all_variables(self):
        """
        Returns
        -------
        The operation to initialize all variables.
        """
        return tf.global_variables_initializer()
            
    @staticmethod
    def create_summaries(var, name):
        """
        Attach several different summaries per variable tensor pased to it.
        Includes the mean, standard deviation, max and minimum values per 
        variable.
        
        Parameters
        ----------
        var : tensorflow.Variable
            Variable tensor to summarize.
        
        name : str
            Name of the summary."
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

    def specify_device(self, device):
        if device == "cpu":
            d = '/cpu:0'
        else:
            d = '/device:GPU:0'
        return d
