import nltk
import pandas as pd
import numpy as np
from preprocess import preprocess

UNK = '<UNK>'
class UnigramModel():
  def set_training_corpus(self, corpus, vocab):
    processed_text  = preprocess(corpus)
    tokens  = nltk.word_tokenize(processed_text)
    
    # Keep only alphabetic tokens and replace OOV words with <UNK>
    self.corpus = [token if token in vocab else UNK for token in tokens if token.isalpha()]

  def load(self):
    # This just initializes the pandas dataframe to keep track of all unigram information (unigram, number of appearences, and probability of occurence)
    self._init_unigram()
    self._init_unigram_count()
    self._init_unigram_probability()

  def run(self, text, vocab):
      """
      Compute the probability of the input text (as product of unigram probabilities).
      """
      preprocessed_text = preprocess(text)
      tokens = nltk.word_tokenize(preprocessed_text)
      # Keep only alphabetic tokens and replace OOV words with <UNK>
      tokens = [token if token in vocab else UNK for token in tokens if token.isalpha()]
      print("Tokens:", tokens)
      word_probabilities = []
      for word in tokens:
          try:
              word_probability = self.df.loc[self.df['unigram'] == word, 'probability'].values[0]
          except IndexError:
              word_probability = 1e-6  # fallback for unseen words
          word_probabilities.append(word_probability)
      return np.prod(word_probabilities)

  def compute_perplexity(self, text, vocab):
      """
      Compute the perplexity of the input text.
      """
      preprocessed_text = preprocess(text)
      tokens = nltk.word_tokenize(preprocessed_text)
      tokens = [token if token in vocab else UNK for token in tokens if token.isalpha()]
      N = len(tokens)
      total_log_prob = 0.0
      for word in tokens:
          try:
              p = self.df.loc[self.df['unigram'] == word, 'probability'].values[0]
          except IndexError:
              p = 1e-6
          total_log_prob += -np.log(p)
      avg_log_prob = total_log_prob / N
      perplexity = np.exp(avg_log_prob)
      return perplexity
  
  def _init_unigram(self):
    # adds all unique words from the corpus into a dataframe
    self.df = pd.DataFrame({'unigram': list(set(self.corpus))})

  def _init_unigram_count(self):
    # gets word count of each word in the corpus
    self.df['count'] = self.df['unigram'].apply(self.corpus.count)

  def _init_unigram_probability(self):
    # calculates the probability of each word in the corpus
    self.df['probability'] = self.df['unigram'].apply(self._calc_unigram_probability)

  def _calc_unigram_probability(self, unigram):
    unigram_count = self.df.loc[self.df['unigram'] == unigram, 'count'].values[0]
    return unigram_count / len(self.corpus)