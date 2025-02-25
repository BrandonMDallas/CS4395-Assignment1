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

def run_unigram_model():
  train_data = Data(TRAIN_PATH).get_contents()
  val_data = Data(VAL_PATH).get_contents()
  unigram = UnigramModel()
  unigram.set_training_corpus(train_data)
  unigram.load()
  result = unigram.run("was no door.")
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
  print(bigram.run("no door was"))
  print("Bigram Model Perplexity: ",bigram_perplexity)
  
def main():
    run_unigram_model()
    #run_bigram_model()
    
if __name__=="__main__":
    main()