import nltk
import pandas as pd
import numpy as np
from preprocess import preprocess
class UnigramModel():
  def set_training_corpus(self, corpus):
    preprocessed_corpus = preprocess(corpus)
    preprocessed_corpus = nltk.word_tokenize(preprocessed_corpus)
    self.corpus = [token for token in preprocessed_corpus if token.isalpha()]
    '''
    
    #self.corpus = corpus
    
    # Tokenize the text in the corpus with a space as a delimeter
    self.corpus = nltk.word_tokenize(self.corpus)
    # Remove all non-alphanumeric characters and make all words lowercase for uniformity in training
    self.corpus = [token.lower() for token in self.corpus if token.isalpha()]
    '''

  def load(self):
    # This just initializes the pandas dataframe to keep track of all unigram information (unigram, number of appearences, and probability of occurence)
    self._init_unigram()
    self._init_unigram_count()
    self._init_unigram_probability()

  def run(self, text):
    # Preprocess the text: lowercases and removes punctuation
    preprocessed_text = preprocess(text)
    # Tokenize the preprocessed text
    tokens = nltk.word_tokenize(preprocessed_text)
    # Filter to keep only alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # For debugging purposes, you can print the tokens
    print(tokens)
    
    # Retrieve the unigram probability for each token in the input text
    word_probabilities = []
    for word in tokens:
        try:
            word_probability = self.df.loc[self.df['unigram'] == word, 'probability'].values[0]
        except IndexError:
            # Use a fallback probability for unseen words
            word_probability = 1e-6
        word_probabilities.append(word_probability)
    
    # Multiply all probabilities together as the final result
    return np.prod(word_probabilities)

  def compute_perplexity(self, text):
    preprocessed_text = preprocess(text)
    tokens = nltk.word_tokenize(preprocessed_text)
    tokens = [token for token in tokens if token.isalpha()]
    N = len(tokens)
    total_log_prob = 0.0
    for word in tokens:
        try:
            p = self.df.loc[self.df['unigram'] == word, 'probability'].values[0]
        except IndexError:
            p = 1e-6  # fallback probability if the word is unseen
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