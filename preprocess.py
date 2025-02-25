import string
import nltk

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