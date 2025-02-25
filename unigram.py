import nltk
import pandas as pd
import numpy as np

class UnigramModel():
  def set_training_corpus(self, corpus):
    self.corpus = corpus
    # Tokenize the text in the corpus with a space as a delimeter
    self.corpus = nltk.word_tokenize(self.corpus)
    # Remove all non-alphanumeric characters and make all words lowercase for uniformity in training
    self.corpus = [token.lower() for token in self.corpus if token.isalpha()]

  def load(self):
    # This just initializes the pandas dataframe to keep track of all unigram information (unigram, number of appearences, and probability of occurence)
    self._init_unigram()
    self._init_unigram_count()
    self._init_unigram_probability()

  def run(self, text):
    # tokenizes the input text to match the tokenization format of the corpus
    text = nltk.word_tokenize(text)
    text = [word.lower() for word in text if word.isalpha()]
    # find the unigram probability for each word in the input text
    word_probabilities = []
    for word in text:
      word_probability = self.df.loc[self.df['unigram'] == word, 'probability'].values[0]
      word_probabilities.append(word_probability)
    # multiply all probabilities together as a final result
    return np.prod(word_probabilities)

  def compute_perplexity(self, text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    N = len(tokens)
    total_log_prob = 0.0
    for word in tokens:
      try:
        p = self.df.loc[self.df['unigram'] == word, 'probability'].values[0]
      except IndexError:
        p = 1e-6  #fallback probability if the bigram was not seen
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