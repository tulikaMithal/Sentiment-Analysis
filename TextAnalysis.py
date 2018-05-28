# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:36:16 2018

@author: lenovo
"""
import re
from collections import Counter

class TextAnalysis:
    
    # We need a function that will split the text based upon sentiment
    def get_text(self,dataset, score):
      # Join together the text in the reviews for a particular sentiment.
      # We lowercase to avoid "Not" and "not" being seen as different words, for example.
       
        s = ""
        for index,row in dataset.iterrows():
            if row['Sentiment'] == score:
                s = s + row['Sentence'].lower()
        
        return s
    
    
    # We also need a function that will count word frequency for each sample
    def count_text(self,text):
      # Split text into words based on whitespace.  Simple but effective.
      words = re.split("\s+", text)
      # Count up the occurence of each word.
      return Counter(words)
  
    
    # We need this function to calculate a count of a given classification
    def get_y_count(self,train_data,score):
      # Compute the count of each classification occuring in the data.
      # return len([r for r in reviews if r[1] == str(score)])
        c = 0
        for index,row in train_data.iterrows():
            if row['Sentiment'] == score:
                c = c + 1
        
        return c