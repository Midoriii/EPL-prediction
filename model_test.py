import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB



#Main fuction for now
if __name__== "__main__":

    for N in [3,5,8]:

        print("Form over last {} matches:\n".format(N))

        data = pd.DataFrame()
        data = data.append(pd.read_csv('data/season14-15/data_with_features_1415_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))
        data = data.append(pd.read_csv('data/season15-16/data_with_features_1516_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))
        data = data.append(pd.read_csv('data/season16-17/data_with_features_1617_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))
        data = data.append(pd.read_csv('data/season17-18/data_with_features_1718_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))

        data_normalized = pd.DataFrame()
        data_normalized = data_normalized.append(pd.read_csv('data/season14-15/data_with_features_1415_norm_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))
        data_normalized = data_normalized.append(pd.read_csv('data/season15-16/data_with_features_1516_norm_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))
        data_normalized = data_normalized.append(pd.read_csv('data/season16-17/data_with_features_1617_norm_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))
        data_normalized = data_normalized.append(pd.read_csv('data/season17-18/data_with_features_1718_norm_form' + str(N) + '.csv', encoding='utf-8', index_col='match_id'))

        #print(data)

        #Get the column names of features, we don't want the result here
        col_names_features = list(data.columns)
        col_names_features.remove('result_home')
        #This will be the Y, which we'll predict
        col_names_outcome = 'result_home'

        #print(col_names_features)

        # Split dataset in training and test datasets
        X_train, X_test = train_test_split(data, test_size=0.15, random_state=int(time.time()))
        # Split normalized dataset in training and test datasets
        Y_train, Y_test = train_test_split(data_normalized, test_size=0.15, random_state=int(time.time()))

        #Create models
        gnb = GaussianNB()
        mnb = MultinomialNB()

        models = [gnb, mnb]
        names_of_models = ["Gaussian NB", "Multinomial NB"]

        # Run predictions on data and normalized data for every model
        for model, name in zip(models, names_of_models):

            # Train classifier
            model.fit(
                X_train[col_names_features].values,
                X_train[col_names_outcome]
            )
            y_pred = model.predict(X_test[col_names_features])

            print("{}: Number of mistakes made out of a total {} predictions : {}, performance {:05.2f}%"
              .format(
                  name,
                  X_test.shape[0],
                  (X_test[col_names_outcome] != y_pred).sum(),
                  100*(1-(X_test[col_names_outcome] != y_pred).sum()/X_test.shape[0])
            ))

            # On normalized data
            model.fit(
                Y_train[col_names_features].values,
                Y_train[col_names_outcome]
            )
            y_pred = model.predict(Y_test[col_names_features])

            print("{} Normalized: Number of mistakes made out of a total {} predictions : {}, performance {:05.2f}%"
              .format(
                  name,
                  Y_test.shape[0],
                  (Y_test[col_names_outcome] != y_pred).sum(),
                  100*(1-(Y_test[col_names_outcome] != y_pred).sum()/Y_test.shape[0])
            ))

            print("\n")
