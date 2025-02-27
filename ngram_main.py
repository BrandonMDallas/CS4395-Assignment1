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

def run_unigram_model(k=1):
    # Load training and validation data
    train_data = Data(TRAIN_PATH).get_contents()
    val_data = Data(VAL_PATH).get_contents()

    # Build vocabulary from the training data
    train_vocab = build_vocab(TRAIN_PATH, min_freq=2)

    # Initialize and train the Unigram model with adjustable smoothing parameter k
    unigram = UnigramModel(k=k)
    unigram.set_training_corpus(train_data, train_vocab)
    unigram.load()

    # Evaluate on an example sentence and print results with k setting
    result = unigram.run("was no door.", train_vocab)
    print(f"Unigram probability product (k={k}):", result)

    unigram_perplexity = unigram.compute_perplexity(val_data, train_vocab)
    print(f"Unigram Model Perplexity (k={k}):", unigram_perplexity)

def run_bigram_model(k=1):
    # Load training and validation data
    train_data = Data(TRAIN_PATH).get_contents()
    val_data = Data(VAL_PATH).get_contents()

    # Build vocabulary from the training data
    train_vocab = build_vocab(TRAIN_PATH, min_freq=2)

    # Initialize and train the Bigram model with adjustable smoothing parameter k
    bigram = BigramModel(k=k)
    bigram.set_training_corpus(train_data, train_vocab)
    bigram.set_val_corpus(val_data, train_vocab)
    bigram.load()

    # Evaluate on an example sentence and print results with k setting
    bigram_result = bigram.run("no door was")
    print(f"Bigram probability product (k={k}):", bigram_result)

    bigram_perplexity = bigram.compute_perplexity(val_data)
    print(f"Bigram Model Perplexity (k={k}):", bigram_perplexity)

  
def main():
    run_unigram_model(0.1)
    run_bigram_model(1)
    
if __name__=="__main__":
    main()