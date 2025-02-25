import nltk
import pandas as pd
import numpy as np
import string

#nltk.download('punkt_tab')

class BigramModel():
  def set_training_corpus(self, corpus):
    sentences = nltk.sent_tokenize(corpus)
    tagged_sentences = []
    for sentence in sentences:
      # removes punctuation
      sentence = sentence.translate(str.maketrans('', '', string.punctuation))
      words = nltk.word_tokenize(sentence)
      words = [word.lower() for word in words]
      if not words:
        continue
      sentence = ['<s>'] + words + ['</s>']
      sentence = ' '.join(sentence)
      tagged_sentences.append(sentence)
    self.corpus = ' '.join(tagged_sentences)

  def set_val_corpus(self, val):
    sentences = nltk.sent_tokenize(val)
    tagged_sentences = []
    for sentence in sentences:
      # removes punctuation
      sentence = sentence.translate(str.maketrans('', '', string.punctuation))
      words = nltk.word_tokenize(sentence)
      words = [word.lower() for word in words]
      if not words:
        continue
      sentence = ['<s>'] + words + ['</s>']
      sentence = ' '.join(sentence)
      tagged_sentences.append(sentence)
    self.val_corpus = ' '.join(tagged_sentences)

  def load(self):
    self._init_bigrams()
    self._init_bigram_counts()
    self._init_bigram_probability()

  def run(self, text):
    sentences = nltk.sent_tokenize(text)
    tagged_sentences = []
    for sentence in sentences:
      # removes punctuation
      sentence = sentence.translate(str.maketrans('', '', string.punctuation))
      words = nltk.word_tokenize(sentence)
      words = [word.lower() for word in words]
      if not words:
        continue
      sentence = ['<s>'] + words + ['</s>']
      sentence = ' '.join(sentence)
      tagged_sentences.append(sentence)
    tokens = (' '.join(tagged_sentences)).split()
    print(tokens)

    """
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = ["<s>"] + tokens
    """
    probabilities = []

    for i in range(1, len(tokens)):
        bigram = (tokens[i-1], tokens[i])
        # Retrieve probability;
        try:
            p = self.df.loc[self.df['bigram'] == bigram, 'probability'].values[0]
        except IndexError:
            p = 1e-6  # fallback probability for unseen bigrams
        probabilities.append(p)

    return np.prod(probabilities)

  def compute_perplexity(self, text):
      #Split validation text into sentences
      sentences = nltk.sent_tokenize(text)
      total_log_prob = 0.0
      # num of bigrams
      total_tokens = 0

      for sentence in sentences:
        #Tokenize & Normalize sentences in validation data
        tokens = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = ["<s>"] + tokens + ["</s>"]

        total_tokens += (len(tokens) - 1)

        # Calculate negative log probability for each bigram
        for i in range(1, len(tokens)):
            bigram = (tokens[i-1], tokens[i])
            try:
                p = self.df.loc[self.df['bigram'] == bigram, 'probability'].values[0]
            except IndexError:
                p = 1e-6  # fallback probability for unseen bigrams
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


""" 
bigram = BigramModel()
bigram.set_training_corpus(get_train())
bigram.set_val_corpus(get_val())
bigram.load()
print(bigram.df)
print(bigram.run("room with big windows. we had wonderful food."))
"""