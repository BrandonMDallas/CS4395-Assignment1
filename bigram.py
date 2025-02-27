import nltk
import pandas as pd
import numpy as np
import string
from collections import Counter
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
    def __init__(self, k=1):
        """
        Initialize the BigramModel with an adjustable smoothing parameter k.
        """
        self.k = k

    def set_training_corpus(self, corpus, vocab):
        tokenized_text = tokenize_sentences(corpus)
        tokens = tokenized_text.split()
        self.corpus = ' '.join([replace_token(token, vocab) for token in tokens])
        self.vocab = vocab
        
    def set_val_corpus(self, val, vocab):
        tokenized_text = tokenize_sentences(val)
        tokens = tokenized_text.split()
        self.val_corpus = ' '.join([replace_token(token, vocab) for token in tokens])

    def load(self):
        self._init_bigrams()
        self._init_bigram_probability()

    def run(self, text):
        preprocessed_text = tokenize_sentences(text)
        tokens = preprocessed_text.split()
        tokens = [replace_token(token, self.vocab) for token in tokens]
        probabilities = []
        for i in range(1, len(tokens)):
            bigram = (tokens[i-1], tokens[i])
            p = self.probability_map.get(bigram, 1e-6)
            probabilities.append(p)
        return np.prod(probabilities)

    def compute_perplexity(self, text):
        preprocessed_text = tokenize_sentences(text)
        tokens = preprocessed_text.split()
        tokens = [replace_token(token, self.vocab) for token in tokens]
        total_log_prob = 0.0
        total_tokens = len(tokens) - 1  # number of bigrams
        for i in range(1, len(tokens)):
            bigram = (tokens[i-1], tokens[i])
            p = self.probability_map.get(bigram, 1e-6)
            total_log_prob += -np.log(p)
        avg_log_prob = total_log_prob / total_tokens
        perplexity = np.exp(avg_log_prob)
        return perplexity

    def _init_bigrams(self):
        tokens = self.corpus.split()
        self.bigrams = list(zip(tokens[:-1], tokens[1:]))
        # Precompute bigram counts and first-word counts using Counter
        self.bigram_counts = Counter(self.bigrams)
        self.first_word_counts = Counter(bigram[0] for bigram in self.bigrams)
        # Optional: create a DataFrame to view counts (not used for inference)
        self.df = pd.DataFrame(list(self.bigram_counts.items()), columns=['bigram', 'count'])

    def _init_bigram_probability(self):
        # Use the adjustable smoothing parameter self.k here
        vocab_size = len(self.vocab)
        self.probability_map = {}
        for bigram, count in self.bigram_counts.items():
            first_word = bigram[0]
            preceding_count = self.first_word_counts[first_word]
            probability = (count + self.k) / (preceding_count + self.k * vocab_size)
            self.probability_map[bigram] = probability

    def _calc_bigram_probability(self, bigram):
        """
        Helper function if you need to calculate probability for a single bigram.
        """
        first_word = bigram[0]
        preceding_count = self.first_word_counts[first_word]
        return (self.bigram_counts[bigram] + self.k) / (preceding_count + self.k * len(self.vocab))
