import numpy as np
import html
import re
import string


def clean_data(data, remove_retweets, remove_mentioner):
    """
    cleans the input data and returns a list of cleaned up texts

    Returns
    -------
    cleaned list of texts
    """
    # option to remove retweets and collect indexes
    if remove_retweets:
        clean_ids = [i[0] for i in enumerate(list(data)) if not str(i[1]).startswith("RT")]
        data = np.array([row for row in list(data) if not str(row).startswith("RT")])
    else:
        clean_ids = [i[0] for i in enumerate(list(data))]

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
    """
    pads list with zeros according to sequence_len

    Parameters
    ----------
    tensors: list
      tensor list to be padded

    sequence_len: int
      the length of each sequence should be. Defaults to the max length
      in the list of sequences.

    Returns
    -------
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