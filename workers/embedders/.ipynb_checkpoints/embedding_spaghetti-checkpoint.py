import re
import string
import html
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from collections import Counter

class EmbeddingSpaghetti(object):
    def __init__(self, data, stopwords_file= None, remove_mentioner = False, remove_retweets = False):
        
        data = pd.read_csv(data)
        
        self.data = data.as_matrix(columns=['SentimentText'])[:, 0]
        self._stopwords_file = stopwords_file
        self._remove_mentioner = remove_mentioner
        self._remove_retweets = remove_retweets
        self.cleaned_data = self._clean_data()
        self.vocab_counts = Counter([item for sublist in self.cleaned_data for item in sublist.split()])
        
        self.embedding_model = None
        self.embeddings = None
        self.tensors_unpadded = None
        self.tensors = None
        self.vocab = None
        self.vocab_size = None
        self._inner_max_length = None
        self._tensor_lengths = None
        
        
        
        
    def _clean_data(self):
        """
        cleans the input data and returns a list of cleaned up texts
        
        Returns
        ----
        cleaned list of texts
        """
        data = self.data
        
        if self._remove_retweets:
            data = np.array([row for row in list(data) if not str(row).startswith("RT")])
        
        # Prepare regex patterns
        ret = []
        reg_punct = '[' + re.escape(''.join(string.punctuation)) + ']' 
        
        # Read in stopwords if file exists
        if self._stopwords_file is not None:
            stopwords = self.__read_stopwords()
            sw_pattern = re.compile(r'\b(' + '|'.join(stopwords) + r')\b')
            
        for row in data:
            # Restore HTML characters
            text = html.unescape(str(row))
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
            
            
            # Remove @ in front of mention if remove_mentioner is <False>
            if not self._remove_mentioner:
                text = re.sub('@', '', text)
            
            words = text.split()
            
            # Remove entire mention including mentioner if remove_mentioner is <True>
            if self._remove_mentioner:
                words = [word for word in words if not word.startswith('@')]
            
            text = ' '.join(words)

            # Transform to lowercase
            text = text.lower()

            # Remove punctuation symbols
            text = re.sub(reg_punct, ' ', text)

            # Replace CC(C+) (a character occurring more than twice in a row) for C
            text = re.sub(r'([a-z])\1{2,}', r'\1', text)

            # Remove stopwords
            if self._stopwords_file is not None:
                text = sw_pattern.sub('', text)
                
            ret += [text]
            
        return ret
    
    def remove_most_common_words(self,prop = 0.02):
        """
        removes the given proportion's most common words from the cleaned data
         
        Parameters
        ----
        prop: float
          the proportion of the most common words to be removed
          
        Returns
        ----
        a list of cleaned texts with the most common words remove. 
        Updates self.cleaned_data
        """
        vocab = list(self.vocab_counts.keys())
    
        remove_num = round(len(vocab) * prop)
    
        removed = self.vocab_counts.most_common(remove_num)
        removed = [item[0] for item in removed]
    
        clean = [phrase.split() for phrase in self.cleaned_data]
    
        new_clean = []
        for phrase in clean:
            p = [word for word in phrase if word not in removed]
            new_clean.append(" ".join(p))
            
        self.cleaned_data = new_clean
        self.vocab_counts = Counter([item for sublist in new_clean for item in sublist.split()])
        
            
        
    def embed(self, embedding_dim = 100, window = 5, min_words = 1, sequence_length = 30):
        """
        creates an embedding layer model
        
        Parameters
        ----
        embedding_dim: int
          the length of the embedding dimesion for each word
          
        window: int
          the number of words to be considered when constructing
          the word embedding
          
        min_words: int
          the minimum number of a words presence in a corpus in
          order to be considered for a word embedding
          
        sequence_len: int
          the number of words per sequence
          
        Returns
        ----
        an embedding model that includes:
          * self.embedding_model - the embedding Word2Vec model object
          * self.embeddings - word embeddings
          * self.tensors_unpadded - a list of sequence indexes that point toward
                                    the word embeddings without padding to a uni-
                                    form sequence length
          * self.vocab - a list of the vocab
          * self.vocab_size - the length of the vocab
          * self.tensors - a list of sequence indexes that point toward the word 
                           embeddings
        """
        text = self.cleaned_data

        # create a word2Vec model
        model = Word2Vec([row.split() for row in text], size = embedding_dim, window = window,min_count=min_words)

        vocab = list(model.wv.vocab)
        word_vectors = [model[x] for x in vocab]

        vocab = [""] + vocab # make first index a blank character for padding purposes
        word_vectors = [np.zeros(embedding_dim,dtype=np.float32)] + word_vectors
        word_vectors = np.array(word_vectors)

        tensors = []
        for row in text:
            tensor_phrase = [vocab.index(x) for x in row.split() if x in vocab]
            tensors.append(tensor_phrase)
        
        tensor_lengths = [len(t) for t in tensors]

        self.embedding_model = model
        self.embeddings = word_vectors 
        self.tensors_unpadded = tensors
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._inner_max_length, self.tensors = self.__pad_with_zeros(tensors, sequence_len = sequence_length)
        self._tensor_lengths = tensor_lengths
        
    

    #~~~~~~~~~~~~~#
    #   HELPERS   #
    #~~~~~~~~~~~~~#
    
    # read in stopwords file if it exists
    def __read_stopwords(self):
        """
        Returns
        ----
        stopwords: list
          a list of stopwords
        """
        if self._stopwords_file is None:
            return None
        with open(self._stopwords_file, mode='r') as f:
            stopwords = f.read().splitlines()
        return stopwords
    
    def __pad_with_zeros(self, lst, sequence_len=None):
        """
        pads list with zeros according to sequence_len
        
        Parameters
        ----
        lst: list
          list to be padded
        
        sequence_len: int
          the length of each sequence should be. Defaults to the max length
          in the list of sequences.
          
        Returns 
        ----
        padding_length used and numpy array of padded tensors.
        """
        # Find maximum length m and ensure that m>=sequence_len
        inner_max_len = max(map(len, lst))
        if sequence_len is not None:
            if inner_max_len > sequence_len:
                raise Exception('Error: Provided sequence length is not sufficient')
            else:
                inner_max_len = sequence_len

        # Pad list with zeros
        result = np.zeros([len(lst), inner_max_len], np.int32)
        for i, row in enumerate(lst):
            for j, val in enumerate(row):
                result[i][j] = val
        return inner_max_len, np.array(result)

    
