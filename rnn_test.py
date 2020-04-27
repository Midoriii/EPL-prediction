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

    scores_1 = []
    scores_2 = []
    scores_3 = []
    accs_1 = []
    accs_2 = []
    accs_3 = []

    for series_length in [3,4,5,8]:

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
        model_1.add(Dense(n_outputs, activation='softmax'))
        model_1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'] )

        # Deeper model
        model_2 = Sequential()
        model_2.add(LSTM(150, input_shape=(series_length, 59), return_sequences=True, dropout=0.4, recurrent_dropout=0.4))
        model_2.add(LSTM(100, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))
        model_2.add(LSTM(50, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))
        model_2.add(LSTM(25, dropout=0.5, recurrent_dropout=0.4))
        model_2.add(Dense(n_outputs, activation='softmax'))
        model_2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'] )

        # Bidirectional model
        model_3 = Sequential()
        model_3.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4), input_shape=(series_length, 59)))
        #model.add(Dropout(0.5))
        model_3.add(Dense(n_outputs, activation='softmax'))
        model_3.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'] )

        print(model_1.summary())
        print(model_2.summary())
        print(model_3.summary())

        # Train for a few epochs
        history_1 = model_1.fit(data_train, labels_train, batch_size=batch_s, epochs=50, validation_split=0.2, verbose=1)
        history_2 = model_2.fit(data_train, labels_train, batch_size=batch_s, epochs=50, validation_split=0.2, verbose=1)
        history_3 = model_3.fit(data_train, labels_train, batch_size=batch_s, epochs=50, validation_split=0.2, verbose=1)

        # Finally make a prediction
        score, acc = model_1.evaluate(data_test, labels_test, batch_size=batch_s)
        scores_1.append(score)
        accs_1.append(acc)

        score, acc = model_2.evaluate(data_test, labels_test, batch_size=batch_s)
        scores_2.append(score)
        accs_2.append(acc)

        score, acc = model_3.evaluate(data_test, labels_test, batch_size=batch_s)
        scores_3.append(score)
        accs_3.append(acc)

    with open("eval/rnn_results.txt", "w") as f:
        f.write("Model 1 score: " + str(scores_1) + "\n" +
              "Model 2 score: " + str(scores_2) + "\n" +
              "Model 3 score: " + str(scores_3) + "\n")
        f.write("Model 1 acc: " + str(accs_1) + "\n" +
              "Model 2 acc: " + str(accs_2) + "\n" +
              "Model 3 acc: " + str(accs_3))

    print("Model 1 score: " + str(scores_1) + "\n" +
          "Model 2 score: " + str(scores_2) + "\n" +
          "Model 3 score: " + str(scores_3))
    print("Model 1 acc: " + str(accs_1) + "\n" +
          "Model 2 acc: " + str(accs_2) + "\n" +
          "Model 3 acc: " + str(accs_3))
