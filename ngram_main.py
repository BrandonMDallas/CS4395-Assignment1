import math
import string
import numpy as np
import pandas as pd
import nltk
from unigram import UnigramModel
from bigram import BigramModel
from data import Data

nltk.download('punkt_tab')
TRAIN_PATH = './A1_DATASET/train.txt'
VAL_PATH = './A1_DATASET/val.txt'
    
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

def run_unigram_model():
  train_data = Data(TRAIN_PATH).get_contents()
  val_data = Data(VAL_PATH).get_contents()
  unigram = UnigramModel()
  unigram.set_training_corpus(train_data)
  unigram.load()
  result = unigram.run("<s>was no door</s>")
  print(result)
  unigram_perplexity = unigram.compute_perplexity(val_data)
  print("Unigram Model Perplexity: ",unigram_perplexity)

def run_bigram_model():
  train_data = Data(TRAIN_PATH).get_contents()
  val_data = Data(VAL_PATH).get_contents()
  bigram = BigramModel()
  bigram.set_training_corpus(train_data)
  bigram.load()
  bigram_perplexity = bigram.compute_perplexity(val_data)
  print(bigram.run("<s>no door was</s>"))
  print("Bigram Model Perplexity: ",bigram_perplexity)
  #bigram = BigramModel()
  #bigram.set_training_corpus(train_data)
  #bigram.load()
  #bigram_perplexity = bigram.compute_perplexity(val_data)
  #print(bigram.run("<s>no door was</s>"))
  #print("Bigram Model Perplexity: ",bigram_perplexity)
  
def main():
    run_unigram_model()
    run_bigram_model()
    #print(tokenize_sentences("I ran up the stairs. It took me a while I was out of breath."))
    #run_unigram_model()
    #run_bigram_model()
    #print("Main")
    
if __name__=="__main__":
    main()