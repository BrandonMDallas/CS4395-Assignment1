import math
import numpy as np
import pandas as pd
import nltk

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



def main():
    print("Main")
    
if __name__=="__main__":
    main()