# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:36:16 2018

@author: lenovo
"""
from collections import Counter
import re

class Predictions:
    
    # Finallt, we create a function that will, given a text example, allow us to calculate the probability
    # of a positive or negative review
    
    def make_class_prediction(self,text, counts, class_prob, class_count):
      prediction = 1
      text_counts = Counter(re.split("\s+", text))
      for word in text_counts:
          # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
          # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
          # We also smooth the denominator counts to keep things even.
          prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
      # Now we multiply by the probability of the class existing in the documents.
      return prediction * class_prob
