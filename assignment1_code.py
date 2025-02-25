import math
import numpy as np
import pandas as pd
import nltk

"""
Used for loading in the train and validation data in google drive
"""
class Data():
  def __init__(self):
    drive.mount('/content/drive')

  def get_train(self):
    train_path = '/content/drive/My Drive/CS_4395_Assignments/A1/train.txt'
    with open(train_path, 'r') as f:
      return f.read()


  def get_val(self):
    val_path = '/content/drive/My Drive/CS_4395_Assignments/A1/val.txt'
    with open(val_path, 'r') as f:
      return f.read()

data = Data()

train_data = data.get_train()
val_data = data.get_val()


    
def preprocess_line(line, lowercase=True):
    tokens = line.strip().split()
    
    if lowercase:
        tokens = [token.lower() for token in tokens]
        
    return tokens

def main():
    print("Main")
    
if __name__=="__main__":
    main()