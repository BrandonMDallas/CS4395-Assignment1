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
    example = "i booked the room"
    result = unigram.run(example, train_vocab)
    print(f"Unigram probability product (k={k}): ", result)

    unigram_perplexity = unigram.compute_perplexity(val_data, train_vocab)
    print(f"Unigram Model Perplexity (k={k}): ", unigram_perplexity, "\n")

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
    example = "i booked the room"
    bigram_result = bigram.run(example)
    print(f"Bigram probability product (k={k}): ", bigram_result)

    bigram_perplexity = bigram.compute_perplexity(val_data)
    print(f"Bigram Model Perplexity (k={k}): ", bigram_perplexity, "\n")

  
def main():
    print("Unigram Model Benchmarks:")
    print("Testing on the phrase: 'i booked the room' \n")
    run_unigram_model(0.1)
    run_unigram_model(0.5)
    run_unigram_model(1)
    run_unigram_model(1.5)
    run_unigram_model(2)
    run_unigram_model(5)
    run_unigram_model(10)
    print("Bigram Model Benchmarks:")
    print("Testing on the phrase: 'i booked the room' \n")
    run_bigram_model(0.1)
    run_bigram_model(0.5)
    run_bigram_model(1)
    run_bigram_model(1.5)
    run_bigram_model(2)
    run_bigram_model(5)
    run_bigram_model(10)
    
if __name__=="__main__":
    main()