# -*- coding: utf-8 -*-
"""Mulder multi-layer CNN neural network class.

Author: James Bain <jamescbain@mizzou.edu>
Last Edited: 2018-03-09

Builds a convolutional neural network for phrase/sentiment classification tasks using the tensorflow API. Word
embeddings can either be pre-trained or randomly initialized upon network instantiation.
"""
import tensorflow as tf

class CNN(object):
    """Convolutional neural network class for sentiment analysis.
    """
    def __init__(self, num_classes, vocab_size, seq_len, embed_dim,
                 filter_sizes, num_filters, batch_size=None, learning_rate=0.001, word_vectors=None,
                 l2_reg_lambda=0.0, fc_layers=2, activation_func="tanh", random_state=None):

        self.inputs = tf.placeholder(tf.int32, [None, seq_len], name="input")
        self.target = tf.placeholder(tf.float32, [None, num_classes], name="target")
        self.dropout = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self._num_total_filters = num_filters * len(filter_sizes)
        self.l2_loss = tf.constant(0.0)

        self.word_vectors = word_vectors
        self.embeddings = self._embed(word_vectors = self.word_vectors, inputs=self.inputs, batch_size=batch_size, 
                                      seq_len=seq_len, embed_dim=embed_dim, vocab_size=vocab_size)
        self.expanded_embeddings = tf.expand_dims(self.embeddings, -1)
        self.conv_max_pool = self._create_convolution_layer(filter_sizes=filter_sizes, embed_dim=embed_dim,
                                                            num_filters=num_filters, seq_len=seq_len,
                                                            fc_layers=fc_layers, activation_func=activation_func,
                                                            dropout=self.dropout,
                                                            expanded_embeddings=self.expanded_embeddings)
        self.scores = self._get_scores(num_classes=num_classes, h_pool_flat=self.conv_max_pool)
        self.prediction = self._predict(scores=self.scores)
        self.losses = self._losses(self.scores, self.target)
        self.accuracy = self._calculate_accuracy(self.prediction, self.target)
        self.loss = self._calculate_loss(self.losses, l2_reg_lambda=l2_reg_lambda)
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
        with tf.name_scope('word_embeddings'):
            if word_vectors is None:
                word_vectors = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1, seed=1))
            else:
                embeddings = tf.Variable(tf.zeros([batch_size, seq_len, embed_dim]), dtype=tf.float32)
            embeddings = tf.nn.embedding_lookup(word_vectors, inputs)
        return embeddings

    def _create_convolution_layer(self, filter_sizes, embed_dim, num_filters, seq_len, 
                                  fc_layers, activation_func, dropout, expanded_embeddings):
        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                weight = tf.Variable(tf.truncated_normal([filter_size, embed_dim, 1, num_filters], 
                                     stddev=0.1), name="weight")
                bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")
                conv = tf.nn.conv2d(expanded_embeddings,
                                    weight,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="convolution")
                # Apply the non-linearity
                hidden = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(hidden, 
                                        ksize=[1, seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                pooled_outputs.append(pooled)
        # Combine pooled features
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self._num_total_filters])
        pooled_layer = tf.nn.dropout(h_pool_flat, dropout)
        
        # Add second fully connected layer
        if fc_layers == 2:
            weight_fc1 = tf.Variable(
                    tf.random_uniform([self._num_total_filters, self._num_total_filters], -1.0, 1.0),
                    name="weight_fc1")
            bias_fc1 = tf.Variable(tf.constant(0.1, shape=[self._num_total_filters]), name="b_fc1")
            if activation_func == 'tanh':
                pooled_layer = tf.nn.tanh(tf.matmul(h_pool_flat, weight_fc1) + bias_fc1, name = 'tanh_fc2')
            else:
                pooled_layer = tf.nn.relu(tf.matmul(h_pool_flat, weight_fc1) + bias_fc1, name = 'relu_fc2')
        return pooled_layer 

    def _get_scores(self, num_classes, h_pool_flat):
        with tf.name_scope("scores"):
            w = tf.get_variable("weight", shape=[self._num_total_filters, num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bias")
            scores = tf.nn.xw_plus_b(h_pool_flat, w, b, name="scores")
            self.l2_loss += tf.nn.l2_loss(w)
            self.l2_loss += tf.nn.l2_loss(b)
        return scores

    def _predict(self, scores):
        # with tf.name_scope("predictions"):
        #    prediction = tf.argmax(score, 1, name="prediction")
        # return prediction
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
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _losses(self, scores, target):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=target, name='cross_entropy')
        return cross_entropy

    def _calculate_loss(self, losses, l2_reg_lambda):
        """Calculates the loss of the prediction.

        Parameters
        ----------
        losses :

        l2_reg_lambda : float

        Returns
        -------
        The loss of the prediction.
        """
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss
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