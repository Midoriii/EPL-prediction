import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.models import Model
from keras.layers import Input
from keras.layers import Bidirectional
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import os
import sys


# Main fuction for now
if __name__== "__main__":

    batch_s = 8
    n_outputs = 3

    scores = []
    accs = []

    for series_length in [3,4,5]:

        data = np.load('rnn_data/data_' + str(series_length) + '.npy')
        labels = np.load('rnn_data/labels_' + str(series_length) + '.npy')

        #Shuffle the daata and labels randomly (but still matching each other!)
        #indices = np.arange(data.shape[0])
        #np.random.shuffle(indices)

        #data = data[indices]
        #labels = labels[indices]

        # One hot encode the labels for classification purposes
        labels = to_categorical(labels, 3)

        #print(data.shape)
        #print(labels.shape)

        # Divive data and labels into a train split and test split, 9:1
        data_train = data[:int(0.9*len(data))]
        data_test = data[int(0.9*len(data)):]
        labels_train = labels[:int(0.9*len(labels))]
        labels_test = labels[int(0.9*len(labels)):]

        # Simple model
        model_1 = Sequential()
        model_1.add(LSTM(128, input_shape=(series_length, 59), dropout=0.4, recurrent_dropout=0.4))
        #model.add(Dropout(0.5))
        model_1.add(Dense(n_outputs, activation='softmax'))
        model_1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'] )

        print(model_1.summary())

        # Train for a few epochs
        history = model_1.fit(data_train, labels_train, batch_size=batch_s, epochs=100, validation_split=0.2, verbose=1)

        # Finally make a prediction
        score, acc = model_1.evaluate(data_test, labels_test, batch_size=8)

        scores.append(score)
        accs.append(acc)

    print(scores)
    print(accs)
