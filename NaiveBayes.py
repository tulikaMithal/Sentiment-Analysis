# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import re
import pandas as pd
from Predictions import Predictions
from TextAnalysis import TextAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# reading training data from textTrainData.txt
f = open(sys.argv[1],"r",encoding="latin-1") 
col1 = []
col2 = []
count =0
for line in f:
    if count == 0:
       count=1
    elif line[-1] == '\n':
        col1.append((line[0:-3]).strip())
        col2.append(int(line[-2]))
    else:
        col1.append((line[0:-2]).strip())
        col2.append(int(line[-1]))
        
final_list = [col1,col2]
train_data = pd.DataFrame(final_list)
train_data = train_data.transpose()
train_data.columns = ['Sentence', 'Sentiment']


# reading test data from textTextData.txt
f = open(sys.argv[2],"r",encoding="latin-1") 
col1 = []
col2 = []
count =0
for line in f:
    if count == 0:
       count=1
    elif line[-1] == '\n':
        col1.append((line[0:-3]).strip())
        col2.append(int(line[-2]))
    else:
        col1.append((line[0:-2]).strip())
        col2.append(int(line[-1]))
        
final_list = [col1,col2]
test_data = pd.DataFrame(final_list)
test_data = test_data.transpose()
test_data.columns = ['Sentence', 'Sentiment']

obj = TextAnalysis()

negative_train_text = obj.get_text(train_data,0)
positive_train_text = obj.get_text(train_data,1)

# Here we generate the word counts for each sentiment
negative_counts = obj.count_text(negative_train_text)
# Generate word counts for positive tone.
positive_counts = obj.count_text(positive_train_text)

# We need these counts to use for smoothing when computing the prediction.
positive_review_count = obj.get_y_count(train_data,1)
negative_review_count = obj.get_y_count(train_data,0)
print(positive_review_count)
print(negative_review_count)

# These are the class probabilities (we saw them in the formula as P(y)).
prob_positive = positive_review_count / len(train_data)
prob_negative = negative_review_count / len(train_data)

obj = Predictions()

predictions = []

for index,row in test_data.iterrows():
    
    # Compute the negative and positive probabilities.
    negative_prediction = obj.make_class_prediction(row[0], negative_counts, prob_negative, negative_review_count)
    positive_prediction = obj.make_class_prediction(row[0], positive_counts, prob_positive, positive_review_count)
        
    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
        predictions.append(0)
    else:
        predictions.append(1)
        
        
actual = test_data.iloc[:,-1].tolist()

print("\nActual target values of test data(given as input by user) :: ",actual )
print("\nPredicted target values of test data(given as input by user):: ",predictions)

print( "\nAccuracy obtained on test data(given as input by user)  :: ", accuracy_score(actual, predictions))
print( "\nConfusion matrix obtained on test data(given as input by user) :: \n", confusion_matrix(actual, predictions))
        
    



