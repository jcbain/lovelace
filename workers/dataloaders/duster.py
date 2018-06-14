# encoding utf-8
#
#
# MIT License
#
# Copyright (c) 2018 James Bain
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Functions for preprocessing data."""

import html
import re
import string
import numpy as np
from nltk import TweetTokenizer
from collections import Counter


def clean_data(data, remove_retweets, remove_mentioner, remove_stopwords, stopwords_prop=0.01):
    """Cleans the input data and returns a list of cleaned up texts.

    Parameters
    ----------
    data: np.array
        Corpus of text where each phrase is a separate array.

    remove_retweets: bool
        Option to remove those phrases (tweets) that begin with 'RT' indicating that the text is a retweet.

    remove_mentioner: bool
        Option to remove those mentioned (@username) within the text.

    remove_stopwords: bool
        Option to remove the top used words.

    stopwords_prop: float
        Proportion of the the words to be removed if remove_stopwords is True.

    Returns
    ----
    ret, cleaned_id: tuple(list, list)
        Cleaned list of texts and the ids.
    """
    # option to remove retweets and collect indexes
    if remove_retweets:
        clean_ids = [i[0] for i in enumerate(list(data)) if not str(i[1]).startswith("RT")]
        data = np.array([row for row in list(data) if not str(row).startswith("RT")])
    else:
        clean_ids = [i[0] for i in enumerate(list(data))]

    if remove_stopwords:
        data = remove_common_words(data, proportion=stopwords_prop)

    # Prepare regex patterns
    ret = []
    reg_punct = '[' + re.escape(''.join(string.punctuation)) + ']'

    for row in data:
        # Restore HTML characters
        text = html.unescape(str(row))
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

        # Remove @ in front of mention if remove_mentioner is <False>
        if not remove_mentioner:
            text = re.sub('@', '', text)

        words = text.split()

        # Remove entire mention including mentioner if remove_mentioner is <True>
        if remove_mentioner:
            words = [word for word in words if not word.startswith('@')]

        text = ' '.join(words)

        # Transform to lowercase
        text = text.lower()

        # Remove punctuation symbols
        text = re.sub(reg_punct, ' ', text)

        # Replace CC(C+) (a character occurring more than twice in a row) for C
        text = re.sub(r'([a-z])\1{2,}', r'\1', text)

        ret += [text]

    return ret, clean_ids


def create_tensors(text, vocab):
    """Creates tensors for each of the phrases in the corpus given a vocabulary.

    Parameters
    ----------
    text: list
        A list of lists of words.

    vocab: dict
        A dict of vocab words that maps the id to the words

    Returns
    -------
    tuple(np.array, np.array)
        The phrases returned as tensors and the lenghth of each tensor.
    """
    tensors = []
    for row in text:
        tensor_phrase = [list(vocab).index(x) for x in row.split() if x in vocab]
        tensors.append(tensor_phrase)

    tensor_lengths = [len(t) for t in tensors]

    return np.array(tensors), np.array(tensor_lengths)


def remove_fat(tensor_lengths, tensors, sequence_len):
    """Remove the extra indexes from tensors that are too long from the training sequence.

    Parameters
    ----------
    tensor_lengths: np.array
        An array of the lengths of tensors.

    tensors: np.array
        Tensors of the data file.

    sequence_len: int
        Sequence length of training model.

    Returns
    -------
    np.array
        Unpadded tensors with max length of [sequence_len].
    """
    n = tensor_lengths - sequence_len
    new_unpadded = []
    for i in range(len(n)):
        if n[i] > 0:
            new_unpadded.append(tensors[i][:-n[i]])
        else:
            new_unpadded.append(tensors[i])

    return np.array(new_unpadded)


def pad_tensors(tensors, sequence_len=None):
    """Pads tensors with zeros according to [sequence_len].

    Parameters
    ----
    tensors: list
        Tensor list to be padded.

    sequence_len: int
        The length of each sequence should be. Defaults to the max length in the list of sequences.

    Returns
    ----
    np.array
        padding_length used and numpy array of padded tensors.
    """
    # Find maximum length m and ensure that m>=sequence_len
    inner_max_len = max(map(len, tensors))
    if sequence_len is not None:
        if inner_max_len > sequence_len:
            raise Exception('Error: Provided sequence length is not sufficient')
        else:
            inner_max_len = sequence_len

    # Pad list with zeros
    result = np.zeros([len(tensors), inner_max_len], np.int32)
    for i, row in enumerate(tensors):
        for j, val in enumerate(row):
            result[i][j] = val
    return np.array(result)


def remove_common_words(data, proportion):
    """Removes the top words of a sample by a give proportion.

    Parameters
    ----------
    data: np.array
        Corpus of text where each phrase is a separate array.

    proportion: float
        The proportion of words that you would like removed.

    Returns
    -------
    top_words_removed: np.array
        Returns the corpus back with the top words removed.
    """
    tokenizer = TweetTokenizer()

    # tokenize the data
    tokenized_data = []
    for s in data:
        try:
            tokenized_data.append(tokenizer.tokenize(s))
        except TypeError:
            pass

    # flatten and remove punctuation
    tokens = [word.lower() for phrase in tokenized_data for word in phrase]
    tokens = [word for word in tokens if word not in set(string.punctuation)]

    # count token occurences
    token_counts = Counter(tokens)

    # find the number for removal
    n_top = round(len(token_counts.keys()) * proportion)

    top_tokens = [t[0] for t in token_counts.most_common(n_top)]

    top_words_removed = []
    for phrase in tokenized_data:
        top_words_removed.append(" ".join([word for word in phrase if word.lower() not in top_tokens]))

    top_words_removed = np.array(top_words_removed)

    return top_words_removed
