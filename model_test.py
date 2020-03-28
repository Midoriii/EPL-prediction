import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB



#Main fuction for now
if __name__== "__main__":

    data = pd.DataFrame()
    data = data.append(pd.read_csv('data/season14-15/data_with_features_1415.csv', encoding='utf-8', index_col='match_id'))
    data = data.append(pd.read_csv('data/season15-16/data_with_features_1516.csv', encoding='utf-8', index_col='match_id'))
    data = data.append(pd.read_csv('data/season16-17/data_with_features_1617.csv', encoding='utf-8', index_col='match_id'))
    data = data.append(pd.read_csv('data/season17-18/data_with_features_1718.csv', encoding='utf-8', index_col='match_id'))

    #print(data)

    #Get the column names of features, we don't want the result here
    col_names_features = list(data.columns)
    col_names_features.remove('result_home')
    #This will be the Y, which we'll predict
    col_names_outcome = 'result_home'

    #print(col_names_features)

    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=int(time.time()))
    #Create nodel
    gnb = GaussianNB()
    mnb = MultinomialNB()

    # Train classifier
    gnb.fit(
        X_train[col_names_features].values,
        X_train[col_names_outcome]
    )
    y_pred = gnb.predict(X_test[col_names_features])

    print("Gaussian: Number of mistakes made out of a total {} predictions : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test[col_names_outcome] != y_pred).sum(),
          100*(1-(X_test[col_names_outcome] != y_pred).sum()/X_test.shape[0])
    ))

    mnb.fit(
        X_train[col_names_features].values,
        X_train[col_names_outcome]
    )
    y_pred = mnb.predict(X_test[col_names_features])

    print("Multinomial: Number of mistakes made out of a total {} predictions : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test[col_names_outcome] != y_pred).sum(),
          100*(1-(X_test[col_names_outcome] != y_pred).sum()/X_test.shape[0])
    ))
