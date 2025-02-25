import math
import string
import numpy as np
import pandas as pd
import nltk

nltk.download('punkt_tab')
"""
Used for loading in the train and validation data in google drive
"""
TRAIN_PATH = 'train.txt'
VAL_PATH = 'val.txt'

class Data():

  def get_train(self):
    with open(TRAIN_PATH, 'r') as f:
      return f.read()


  def get_val(self):
    with open(VAL_PATH, 'r') as f:
      return f.read()

data = Data()

train_data = data.get_train()
val_data = data.get_val()

    
"""
Preprocessing methods
"""
def preprocess(text):
    text = text.lower()  # Lowercase normalization
    text = text.translate(str.maketrans('', '', string.punctuation)) #Remove punctuation
    return text
  
def tokenize_sentences(text):
  sentences = nltk.sent_tokenize(text)
  tagged_sentences = []
  
  for sentence in sentences:
    preprocessed_sentence = preprocess(sentence)
    words = nltk.word_tokenize(preprocessed_sentence)
    if not words:
      continue
    new_sentence = ['<s>'] + words + ['</s>']
    new_sentence = ' '.join(new_sentence)
    tagged_sentences.append(new_sentence)
  return ' '.join(tagged_sentences)

"""
Class that handles the Unigram Model
"""
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

"""
Class that handles the Bigram Model
"""
class BigramModel():

  def set_training_corpus(self, corpus):
      # Tokenize and normalize the corpus
      self.corpus = nltk.word_tokenize(corpus)
      self.corpus = [token.lower() for token in self.corpus if token.isalpha()]

  def load(self):
    self._init_bigram()
    self._init_bigram_counts()
    self._init_bigram_probability()

  def run(self, text):

      # Tokenize and normalize the text
      tokens = nltk.word_tokenize(text)
      tokens = [token.lower() for token in tokens if token.isalpha()]
      tokens = ["<s>"] + tokens
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

  def _init_bigram(self):

    #Split corpus into sentences
    sentences = nltk.sent_tokenize(" ".join(self.corpus))
    tokens = []

    for sentence in sentences:
        #Tokenize & Normalize each sentence
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_tokens = [token.lower() for token in sentence_tokens if token.isalpha()]

        #Add start & end token on sentences
        tokens.extend(["<s>"] + sentence_tokens + ["</s>"])

    self.tokens = tokens

    #Generate bigram tuples: (prev_word, curr_word)
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
  

def run_unigram_model():
  unigram = UnigramModel()
  unigram.set_training_corpus(train_data)
  unigram.load()
  result = unigram.run("<s>was no door</s>")
  print(result)
  unigram_perplexity = unigram.compute_perplexity(val_data)
  print("Unigram Model Perplexity: ",unigram_perplexity)

def run_bigram_model():
  bigram = BigramModel()
  bigram.set_training_corpus(train_data)
  bigram.load()
  bigram_perplexity = bigram.compute_perplexity(val_data)
  print(bigram.run("<s>no door was</s>"))
  print("Bigram Model Perplexity: ",bigram_perplexity)
  
def main():
    print(tokenize_sentences("I ran up the stairs. It took me a while I was out of breath."))
    #run_unigram_model()
    #run_bigram_model()
    #print("Main")
    
if __name__=="__main__":
    main()