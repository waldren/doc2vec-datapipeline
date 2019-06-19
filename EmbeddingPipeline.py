# Class to create embedding pipeline
#import urllib.request
import tensorflow as tf
from collections import Counter
import string
import numpy as np 
import os
import io
import collections
#import requests
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from string import digits

class EmbeddingPipeline:
  def __init__(self, session, window_size=2, method="doc2vec", stop_words=[], vocabulary_size=100, word_dict={}, process_ct=32):
    self.window_size = window_size
    self.stop_words = stop_words
    self.vocabulary_size = vocabulary_size
    self.word_dict = word_dict
    self.method = method
    self.process_ct = process_ct
    self.session = session

  # New normalization pipeline that replaces the cookbook code
  def normalize(self, text):
    #Strip non letters
    text = tf.strings.regex_replace(text, pattern='[^A-Za-z ]', rewrite='')
    #to lower case
    chars = tf.strings.unicode_decode(text, input_encoding='UTF-8')
    capital_mask = tf.logical_and(tf.greater_equal(chars, 65), tf.less(chars, 91))
    chars = chars + tf.cast(capital_mask, tf.int32) * 32
    text = tf.strings.unicode_encode(chars, output_encoding='UTF-8')
    # remove stop words
    for sw in self.stop_words:
      p = " {0}|{0} ".format(sw)
      text = tf.strings.regex_replace(text, pattern=p, rewrite='')
    text = tf.strings.strip(text)
    return text


    # Code is from cookbook
    # Turn text data into lists of integers from dictionary
  def text_to_numbers(self, sentences):
    # Initialize the returned data
    data = []
    for sentence in sentences:
      sentence_data = []
      # For each word, either use selected index or rare word index
      for word in sentence.split():
        if word in self.word_dict:
          word_ix = self.word_dict[word]
        else:
          word_ix = 0
        sentence_data.append(word_ix)
      data.append(sentence_data)
    return(data)
    
  # Code is from
  # https://stackoverflow.com/questions/35857519/efficiently-count-word-frequencies-in-python
  def build_dictionary(self, sentences):
    # Note that `ngram_range=(1, 1)` means we want to extract Unigrams, i.e. tokens.
    ngram_vectorizer = CountVectorizer(analyzer='word', tokenizer=word_tokenize, ngram_range=(1, 1), min_df=1)
    # X matrix where the row represents sentences and column is our one-hot vector for each token in our vocabulary
    X = ngram_vectorizer.fit_transform(sentences)

    # Vocabulary
    vocab = list(ngram_vectorizer.get_feature_names())

    # Column-wise sum of the X matrix.
    # It's some crazy numpy syntax that looks horribly unpythonic
    # For details, see http://stackoverflow.com/questions/3337301/numpy-matrix-to-array
    # and http://stackoverflow.com/questions/13567345/how-to-calculate-the-sum-of-all-columns-of-a-2d-numpy-array-efficiently
    counts = X.sum(axis=0).A1

    freq_distribution = Counter(dict(zip(vocab, counts)))
    self.word_dict = freq_distribution.most_common(self.vocabulary_size)
    
    return freq_distribution.most_common(self.vocabulary_size)  

  # This is not working and not finished
  def generate_window_sequence(self, sentence_ix, sentence, label):
    
    ws = tf.strings.split([sentence], sep=' ').values

    offset = tf.constant(int(self.window_size/2))
    i = tf.constant(int(self.window_size/2))

    def while_condition (i, window_sequences, offset):
      return tf.less(i, tf.math.subtract(tf.size(ws), offset))
    
    def body(i, window_sequences, offset):
      #TODO - move within office if it is > 1
      before = tf.math.subtract(i, offset)
      after = tf.math.add(i, offset)
      window_sequences.append((tf.gather(ws, [before, i])))
      window_sequences.append((tf.gather(ws, [after, i])))
      return [tf.math.add(i,1), window_sequences, offset]
      
    window_sequences = []
    r = tf.while_loop(while_condition, body, [i, window_sequences, offset])

    return sentence_ix, window_sequences, label

  def build_windows(self, i):
    window_sequences = []
    offset = tf.constant(int(self.window_size/2))
    # TODO - move within offset if < 1
    window_sequences.append((words[i-offset], words[i]))
    window_sequences.append((words[i+offset], words[i]))
    return window_sequences
  
  # This is part of the original cookbook code that I am trying to replace using tf.Data pipeline
  def nofunct(self, x):
    window_sequences = [sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(sentence)]
    label_indices = [ix if ix<window_size else window_size for ix,x in enumerate (window_sequences)]
            # Pull out center word of interest for each window and create a tuple for each window
    if self.method=='skip_gram':
      batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
      # Make it in to a big list of tuples (target word, surrounding word)
      tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
      batch, labels = [list(x) for x in zip(*tuple_data)]
    elif self.method=='cbow':
      batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
      # Only keep windows with consistent 2*window_size
      batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
      batch, labels = [list(x) for x in zip(*batch_and_labels)]
    elif self.method=='doc2vec':
      # For doc2vec we keep LHS window only to predict target word
      batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
      batch, labels = [list(x) for x in zip(*batch_and_labels)]
      # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
      batch = [x + [rand_sentence_ix] for x in batch]
    else:
      raise ValueError('Method {} not implemented yet.'.format(self.method))

    return(batch_data, label_data)
  
  def generate_dataset(self, sentences, labels):
    ds = tf.data.Dataset.from_tensor_slices(sentences)
    ds = ds.map(self.normalize, num_parallel_calls=self.process_ct)
    ds_ix = tf.data.Dataset.from_tensor_slices( tf.range(0, len(sentences)) )
    ds_label = tf.data.Dataset.from_tensor_slices(labels)
    ds = ds.zip((ds_ix,ds,ds_label))
    ds = ds.map(self.generate_window_sequence, num_parallel_calls=self.process_ct)
    return ds

    
