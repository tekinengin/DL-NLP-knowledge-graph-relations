#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import print_function
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

#Read Data
data = pd.read_csv('hw1_train.csv')
test_data = pd.read_csv('hw1_test.csv')

#Extract Unique Labels
lab_col = set()
for x in data['CORE RELATIONS']:
    for y in x.split(' '):
        lab_col.add(y)
        
#Create Label Columns with Zero Value
for x in lab_col:
    data.insert(len(data.columns),x,np.zeros(data.shape[0]))

#Fill in Label Columns with true Labels 
for x,y in data.iterrows():
    for z in y['CORE RELATIONS'].split(' '):
         data.loc[x,z] = 1

#Drop unnecessary columns            
data = data.drop('CORE RELATIONS',axis=1)

# Helper Function to calculate F1-score
# @Credit to stackoverflow.com
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#Shuffle data before training
data = data.sample(frac=1).reset_index(drop=True)
X_train = data['UTTERANCE']
y_train = data.drop('ID',axis=1).drop('UTTERANCE',axis=1)
X_test = test_data['UTTERANCE']

#Create Vocabulary and tokenize data
max_words=1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

batch_size = 128
epochs = 20
num_nodes = 512
dropout = 0.5

#MLP Model with one-hidden layer and 46 output layer
model = Sequential()

model.add(Dense(num_nodes, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(46))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',f1_m])

#Fitting Model
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

#Predictions
y_pred = model.predict(X_test) 

predictions = []

#Converting Predictions to Text 
#We have a special case for NO_REL if we have predicated any other label than NO_REL then no need to use NO_REL
for i in y_pred:
    temp = ''
    for index,p in enumerate(i):
        if(p>0.17 and data.columns[index+2] != 'NO_REL'):    
            temp = temp + ' ' + data.columns[index+2]
    if(len(temp) == 0):
        temp = 'NO_REL'
    predictions.append(temp.strip())
    
idx = [i for i in range(len(predictions))]

#Saving as .csv for submission
df = {'ID': idx, 'CORE RELATIONS': predictions}
df = pd.DataFrame.from_dict(df)
df.to_csv('hw1_pred_MLP.csv',index=False)





