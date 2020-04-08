import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



#Main fuction for now
if __name__== "__main__":

    # Dictionary to store results corresponding to model name
    results_dict = {}

    for N in [3,5,8]:

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

        knn_15 = KNeighborsClassifier(n_neighbors=15)
        knn_25 = KNeighborsClassifier(n_neighbors=25)
        knn_35 = KNeighborsClassifier(n_neighbors=35)

        svm = SVC(gamma='auto')


        models = [gnb, mnb, knn_15, knn_25, knn_35, svm]
        names_of_models = ["Gaussian NB", "Multinomial NB", "KNN-15", "KNN-25", "KNN-35", "SVM"]

        # Run predictions on data and normalized data for every model
        for model, name in zip(models, names_of_models):

            # Train classifier
            model.fit(
                X_train[col_names_features].values,
                X_train[col_names_outcome]
            )
            y_pred = model.predict(X_test[col_names_features])

            # Get the performance/accuraccy in %
            perf1 = 100*(1-(X_test[col_names_outcome] != y_pred).sum()/X_test.shape[0])

            results_dict[name + "  form: " + str(N)] = perf1

            # On normalized data
            model.fit(
                Y_train[col_names_features].values,
                Y_train[col_names_outcome]
            )
            y_pred = model.predict(Y_test[col_names_features])

            perf2 = 100*(1-(Y_test[col_names_outcome] != y_pred).sum()/Y_test.shape[0])

            results_dict[name + "  form: " + str(N) + "  Normalized"] = perf2

    print("\n")
    # Print the accuraccy of each model in descending order
    for model in sorted(results_dict, key=results_dict.get, reverse = True):
        print(model + "   " + str(results_dict[model]))
    print("\n")
