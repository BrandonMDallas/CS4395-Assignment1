import nltk
import pandas as pd
import numpy as np
import string
from preprocess import tokenize_sentences


UNK = '<UNK>'

SPECIAL_TOKENS = {'<s>', '</s>'}

def replace_token(token, vocab):
    """
    Returns the token if it's a special token or if it's in the vocabulary;
    otherwise, returns the UNK token.
    """
    if token in SPECIAL_TOKENS:
        return token
    return token if token in vocab else UNK

class BigramModel():
  def set_training_corpus(self, corpus, vocab):
    tokenized_text = tokenize_sentences(corpus)
    tokens = tokenized_text.split()
    self.corpus = ' '.join([replace_token(token, vocab) for token in tokens])

  def set_val_corpus(self, val, vocab):
      tokenized_text = tokenize_sentences(val)
      tokens = tokenized_text.split()
      self.val_corpus = ' '.join([replace_token(token, vocab) for token in tokens])

  def load(self):
    self._init_bigrams()
    self._init_bigram_counts()
    self._init_bigram_probability()

  def run(self, text, vocab):
    preprocessed_text = tokenize_sentences(text)
    tokens = preprocessed_text.split()
    # Replace tokens using the helper function
    tokens = [replace_token(token, vocab) for token in tokens]
    probabilities = []
    for i in range(1, len(tokens)):
        bigram = (tokens[i-1], tokens[i])
        try:
            p = self.df.loc[self.df['bigram'] == bigram, 'probability'].values[0]
        except IndexError:
            p = 1e-6
        probabilities.append(p)
    return np.prod(probabilities)

  def compute_perplexity(self, text, vocab):
    preprocessed_text = tokenize_sentences(text)
    tokens = preprocessed_text.split()
    tokens = [replace_token(token, vocab) for token in tokens]
    total_log_prob = 0.0
    total_tokens = len(tokens) - 1  # number of bigrams
    for i in range(1, len(tokens)):
        bigram = (tokens[i-1], tokens[i])
        try:
            p = self.df.loc[self.df['bigram'] == bigram, 'probability'].values[0]
        except IndexError:
            p = 1e-6
        total_log_prob += -np.log(p)
    avg_log_prob = total_log_prob / total_tokens
    perplexity = np.exp(avg_log_prob)
    return perplexity

  def _init_bigrams(self):
    tokens = self.corpus.split()
    self.bigrams = list(zip(tokens[:-1], tokens[1:]))
    unique_bigrams = list(set(self.bigrams))
    self.df = pd.DataFrame({'bigram': unique_bigrams})

  def _init_bigram_counts(self):
    self.df['count'] = self.df['bigram'].apply(lambda bg: self.bigrams.count(bg))

  def _init_bigram_probability(self):
    self.df['probability'] = self.df['bigram'].apply(self._calc_bigram_probability)

  def _calc_bigram_probability(self, bigram):
    first_word = bigram[0]
    preceding_count = sum(1 for bg in self.bigrams if bg[0] == first_word)
    if preceding_count == 0:
        return 0
    return self.bigrams.count(bigram) / preceding_count