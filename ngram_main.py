import math
import string
import numpy as np
import pandas as pd
import nltk
from unigram import UnigramModel
from bigram import BigramModel
from data import Data
from vocabulary import build_vocab, replace_unknowns

nltk.download('punkt_tab')
TRAIN_PATH = './A1_DATASET/train.txt'
VAL_PATH = './A1_DATASET/val.txt'

def run_unigram_model():
  train_data = Data(TRAIN_PATH).get_contents()
  val_data = Data(VAL_PATH).get_contents()
  
  # Build vocabulary from the training data (using the file directly)
  train_vocab = build_vocab(TRAIN_PATH, min_freq=2)
  
  #replace_unknowns(VAL_PATH, train_vocab, 'validation_processed.txt')
  
  # Initialize and train the Unigram model using the processed data
  unigram = UnigramModel()
  unigram.set_training_corpus(train_data, train_vocab)
  unigram.load()
  
  # Evaluate the model on an example sentence
  result = unigram.run("was no door.", train_vocab)
  print("Unigram probability product:", result)
  
  unigram_perplexity = unigram.compute_perplexity(val_data, train_vocab)
  print("Unigram Model Perplexity:", unigram_perplexity)

def run_bigram_model():
    # Load training and validation data from file
    train_data = Data(TRAIN_PATH).get_contents()
    val_data = Data(VAL_PATH).get_contents()
    
    # Build vocabulary from the training data (using the file directly)
    train_vocab = build_vocab(TRAIN_PATH, min_freq=2)
    
    # Initialize and train the Bigram model using the processed data
    bigram = BigramModel(k = 1)
    bigram.set_training_corpus(train_data, train_vocab)
    bigram.set_val_corpus(val_data, train_vocab)
    bigram.load()
    
    # Evaluate the model on an example sentence
    bigram_result = bigram.run("no door was")
    print("Bigram probability product:", bigram_result)
    
    bigram_perplexity = bigram.compute_perplexity(val_data)
    print("Bigram Model Perplexity:", bigram_perplexity)
  
def main():
    run_unigram_model()
    run_bigram_model()
    
if __name__=="__main__":
    main()