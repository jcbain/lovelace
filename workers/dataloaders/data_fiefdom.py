import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from workers.dataloaders.duster import *


class DataOverlord(object):
    def __init__(self, data_file, embedding_file, sequence_len=None, test_size=0.2,
                 val_samples=100, remove_retweets=True, remove_mentioner=True, remove_stopwords=False, random_state=0):

        self._input_file = data_file
        self.sequence_len = sequence_len

        self._remove_retweets = remove_retweets
        self._remove_mentioner = remove_mentioner

        # read in the new data file
        data = pd.read_csv(data_file)
        sentiments = np.squeeze(data.as_matrix(columns=['Sentiment']))
        zeros_n_rows = len(sentiments)
        zeros_n_cols = len(pd.unique(sentiments))
        one_hot_array = np.zeros((zeros_n_rows, zeros_n_cols))
        one_hot_array[np.arange(zeros_n_rows), sentiments] = 1

        self._sentiments = one_hot_array
        self.samples = data.as_matrix(columns=['Text'])[:, 0]

        if remove_retweets:
            self._sentiments = np.delete(self._sentiments, self._find_retweet_indexes(), 0)

        self.data, _ = clean_data(data=self.samples, remove_retweets=remove_retweets,
                                  remove_mentioner=remove_mentioner, remove_stopwords=remove_stopwords)
        self.embeddings, self.vocab = self._embed(embedding_file=embedding_file)
        self.vocab_size = len(self.vocab)

        self.tensors_unpadded, self.tensor_lengths = create_tensors(text=self.data, vocab=self.vocab)

        self.tensors = pad_tensors(tensors=self.tensors_unpadded, sequence_len=sequence_len)

        # Split data in train, validation and test sets
        indices = np.arange(len(self._sentiments))
        x_tv, self._x_test, y_tv, self._y_test, tv_indices, test_indices = train_test_split(
            self.tensors,
            self._sentiments,
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self._sentiments[:, 0])
        self._x_train, self._x_val, self._y_train, self._y_val, train_indices, val_indices = train_test_split(
            x_tv,
            y_tv,
            tv_indices,
            test_size=val_samples,
            random_state=random_state,
            stratify=y_tv[:, 0])
        self._val_indices = val_indices
        self._test_indices = test_indices
        self._train_lengths = self.tensor_lengths[train_indices]
        self._val_lengths = self.tensor_lengths[val_indices]
        self._test_lengths = self.tensor_lengths[test_indices]
        self._current_index = 0
        self._epoch_completed = 0

    def _embed(self, embedding_file):
        if embedding_file == "None" or embedding_file is None:
            embeddings = None
            word_list = [word for row in self.data for word in row.split()]
            vocab = []
            for word in word_list:
                if word not in vocab:
                    vocab.append(word)
            vocab = [""] + vocab  # make first index a blank character for padding purposes
        else:
            npzfile = np.load(embedding_file)
            embeddings = npzfile['embeddings']
            vocab = npzfile['vocab']

        return embeddings, vocab

    def _find_retweet_indexes(self):
        """
        finds those samples in a file that begin with 'RT' and therefore a retweet

        Returns
        ----
        a list of indexes of those rows that are retweets
        """
        counter = 0
        removal_indexes = []

        for row in self.samples:
            if str(row).startswith("RT"):
                removal_indexes.append(counter)
            counter += 1

        return removal_indexes

    def next_batch(self, batch_size):
        """
        provides a new batch of rows of the size [batch_size]

        Parameters
        ----
        batch_size: int
          the number of rows to be returned in that batch

        Returns
        ----
        samples of the size [batch_size], which inclues:
          * self._x_train: input tensor
          * self._y_train: target
          * self._train_lengths: lenghts of text
        """
        start = self._current_index
        self._current_index += batch_size
        if self._current_index > len(self._y_train):
            # Complete epoch and randomly shuffle train samples
            self._epoch_completed += 1
            ind = np.arange(len(self._y_train))
            np.random.shuffle(ind)
            self._x_train = self._x_train[ind]
            self._y_train = self._y_train[ind]
            self._train_lengths = self._train_lengths[ind]
            start = 0
            self._current_index = batch_size
        end = self._current_index
        return self._x_train[start:end], self._y_train[start:end], self._train_lengths[start:end]

    def get_val_data(self):
        """
        gets the validation data

        Returns
        -----
        validation data, which includes:
          * self._x_val: input tensors
          * self._y_val: target
          * self._val_lengths: lengths of text
        """
        return self._x_val, self._y_val, self._val_lengths

    def get_test_data(self):
        """
        gets the testing data

        Returns
        -----
        testing data, which includes:
          * self._x_test: input tensors
          * self._y_test: target
          * self._test_lengths: lengths of text
        """
        return self._x_test, self._y_test, self._test_lengths


class DataPleb(object):
    def __init__(self, vocab, sequence_len, data_file=None, indiv_text=None, remove_retweets=True,
                 remove_mentioner=True, text_name='Text', extra_atts=None, encoding=None, sep='|', qchar='&'):
        self.vocab = vocab
        self._text_name = text_name
        self._input_file = data_file
        self._input_phrase = indiv_text
        self.sequence_len = sequence_len
        self._extra_atts = extra_atts

        self._remove_retweets = remove_retweets
        self._remove_mentioner = remove_mentioner

        data = self._data(encoding=encoding, sep=sep, qchar=qchar)
        self.samples = data.as_matrix(columns=[text_name])[:, 0]

        self.data, self.clean_ids = clean_data(data=self.samples, remove_retweets=remove_retweets,
                                               remove_mentioner=remove_mentioner, remove_stopwords=False)
        self.atts = self._atts(data=data)

        self.vocab_size = len(self.vocab)

        tensors_unpadded, self.tensor_lengths = create_tensors(text=self.data, vocab=self.vocab)
        self.tensors_unpadded = remove_fat(tensor_lengths=self.tensor_lengths, tensors=tensors_unpadded,
                                           sequence_len=sequence_len)

        self.tensors = pad_tensors(self.tensors_unpadded, sequence_len=sequence_len)

    def _data(self, encoding, sep, qchar):
        """Initial processing of data.

        Stores inputs in a pd.DataFrame.

        Returns
        -------
        data: pd.DataFrame
        """
        if self._input_file is not None:
            data = pd.read_csv(self._input_file, encoding=encoding, error_bad_lines=False, sep=sep, quotechar=qchar)
        else:
            data = pd.DataFrame([self._input_phrase], columns=[self._text_name])
        return data

    def _atts(self, data):
        """Create a matrix of supplemental atttributes.

        This method is useful for extracting the other attributes that match the row indexes
        of the tensors. This is beneficial for further analysis after prediction has been conducted.

        Parameter
        ---------
        data: pd.DataFrame
            Data that has attributes you want other than just the text feature.

        Returns
        -------
        atts: np.ndarray or None
            Attributes in ndarray format pertaining to the tensors used.
        """
        if (self._input_file is not None and self._extra_atts is not None):
            # check if chosen columns exist
            try:
                atts = data.loc[self.clean_ids, self._extra_atts]
            except KeyError as e:
                print(e)
                atts = data.loc[self.clean_ids, :]
                pass
            atts = atts.astype('object')
            atts = atts.as_matrix()
        else:
            atts = None
        return atts
