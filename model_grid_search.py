import pandas as pd
import numpy as np
import time
import os
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV

import model_test as mt


# Eval the best grid model on test data
def evaluate_grid_model(X_test, Y_test, model):
    # Predict on test data
    y_pred = model.predict(X_test)
    # Return the performance/accuracy in %
    return accuracy_score(Y_test, y_pred)



# Main fuction for now
if __name__== "__main__":

    # Dictionary to store results corresponding to model name
    results_dict = defaultdict(lambda: 0)

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
        #X_train, X_test = train_test_split(data, test_size=0.15, random_state=int(time.time()))

        # Or alternatively try to guess the results of the end of the season
        X_train = data[:320].append(data[370:690]).append(data[740:1060]).append(data[1110:1430])
        X_test = data[320:370].append(data[690:740]).append(data[1060:1110]).append(data[1430:1480])

        # Split normalized dataset in training and test datasets
        #Y_train, Y_test = train_test_split(data_normalized, test_size=0.15, random_state=int(time.time()))

        # Rest of the season testing
        Y_train = data_normalized[:320].append(data_normalized[370:690]).append(data_normalized[740:1060]).append(data_normalized[1110:1430])
        Y_test = data_normalized[320:370].append(data_normalized[690:740]).append(data_normalized[1060:1110]).append(data_normalized[1430:1480])

        # Split data into features and labels
        train_data = X_train[col_names_features]
        train_labels = X_train[col_names_outcome]
        test_data = X_test[col_names_features]
        test_labels = X_test[col_names_outcome]

        train_data_norm = Y_train[col_names_features]
        train_labels_norm = Y_train[col_names_outcome]
        test_data_norm = Y_test[col_names_features]
        test_labels_norm = Y_test[col_names_outcome]

        #Feature extraction - 20 top
        reduced_features_20 = mt.get_reduced_columns(20, train_data, train_labels)
        #Do again for normalized data
        reduced_features_normalized_20 = mt.get_reduced_columns(20, train_data_norm, train_labels_norm)

        reduced_features_30 = mt.get_reduced_columns(30, train_data, train_labels)
        reduced_features_normalized_30 = mt.get_reduced_columns(30, train_data_norm, train_labels_norm)

        reduced_features_40 = mt.get_reduced_columns(40, train_data, train_labels)
        reduced_features_normalized_40 = mt.get_reduced_columns(40, train_data_norm, train_labels_norm)

        reduced_features_50 = mt.get_reduced_columns(50, train_data, train_labels)
        reduced_features_normalized_50 = mt.get_reduced_columns(50, train_data_norm, train_labels_norm)

        reduced_features_60 = mt.get_reduced_columns(60, train_data, train_labels)
        reduced_features_normalized_60 = mt.get_reduced_columns(60, train_data_norm, train_labels_norm)


        forest = RandomForestClassifier()
        extra_forest = ExtraTreesClassifier()

        # Create the parameter grid based for GridSearchCV
        param_grid = {
            'max_depth': [40, 60, 100, 160],
            'max_features': [2, 6, 10],
            'min_samples_leaf': [3, 10, 20],
            'min_samples_split': [6, 10, 16],
            'n_estimators': [100, 500, 1000]
        }

        # Instantiate the grid search model
        grid_search_rf = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = 2)
        grid_search_ef = GridSearchCV(estimator = extra_forest, param_grid = param_grid, cv = 3, n_jobs = 2)

        print("RF Raw " + str(N) + ":")
        grid_search_rf.fit(train_data, train_labels)
        print(grid_search_rf.best_params_)
        results_dict["params RF Raw " + str(N)] = grid_search_rf.best_params_
        results_dict["RF Raw " + str(N)] = evaluate_grid_model(test_data, test_labels, grid_search_rf.best_estimator_)

        print("RF Raw norm " + str(N) + ":")
        grid_search_rf.fit(train_data_norm, train_labels_norm)
        print(grid_search_rf.best_params_)
        results_dict["RF Raw norm " + str(N)] = grid_search_rf.best_params_
        results_dict["RF Raw norm " + str(N)] = evaluate_grid_model(test_data_norm, test_labels_norm, grid_search_rf.best_estimator_)

        print("ExF Raw " + str(N) + ":")
        grid_search_ef.fit(train_data, train_labels)
        print(grid_search_ef.best_params_)
        results_dict["ExF Raw " + str(N)] = grid_search_ef.best_params_
        results_dict["ExF Raw " + str(N)] = evaluate_grid_model(test_data, test_labels, grid_search_ef.best_estimator_)

        print("ExF Raw norm " + str(N) + ":")
        grid_search_ef.fit(train_data_norm, train_labels_norm)
        print(grid_search_ef.best_params_)
        results_dict["ExF Raw norm " + str(N)] = grid_search_ef.best_params_
        results_dict["ExF Raw norm " + str(N)] = evaluate_grid_model(test_data_norm, test_labels_norm, grid_search_ef.best_estimator_)



        features = [reduced_features_20, reduced_features_30, reduced_features_40, reduced_features_50, reduced_features_60]
        features_normalized = [reduced_features_normalized_20, reduced_features_normalized_30, reduced_features_normalized_40, reduced_features_normalized_50, reduced_features_normalized_60]

        descriptions = ["Reduced-20", "Reduced-30", "Reduced-40", "Reduced-50", "Reduced-60"]

        for feats, feats_norm, desc in zip(features, features_normalized, descriptions):
            print("RF Raw " + desc + str(N) + " :")
            grid_search_rf.fit(train_data[feats], train_labels)
            print(grid_search_rf.best_params_)
            results_dict["RF Raw " + desc + str(N)] = grid_search_rf.best_params_
            results_dict["RF Raw " + desc + str(N)] = evaluate_grid_model(test_data[feats], test_labels, grid_search_rf.best_estimator_)

            print("RF Raw norm " + desc + str(N) + " :")
            grid_search_rf.fit(train_data_norm[feats_norm], train_labels_norm)
            print(grid_search_rf.best_params_)
            results_dict["RF Raw norm " + desc + str(N)] = grid_search_rf.best_params_
            results_dict["RF Raw norm " + desc + str(N)] = evaluate_grid_model(test_data_norm[feats_norm], test_labels_norm, grid_search_rf.best_estimator_)

            print("ExF Raw " + desc + str(N) + " :")
            grid_search_ef.fit(train_data[feats], train_labels)
            print(grid_search_ef.best_params_)
            results_dict["ExF Raw " + desc + str(N)] = grid_search_ef.best_params_
            results_dict["ExF Raw " + desc + str(N)] = evaluate_grid_model(test_data[feats], test_labels, grid_search_ef.best_estimator_)

            print("ExF Raw norm " + desc + str(N) + " :")
            grid_search_ef.fit(train_data_norm[feats_norm], train_labels_norm)
            print(grid_search_ef.best_params_)
            results_dict["ExF Raw norm " + desc + str(N)] = grid_search_ef.best_params_
            results_dict["ExF Raw norm " + desc + str(N)] = evaluate_grid_model(test_data_norm[feats_norm], test_labels_norm, grid_search_ef.best_estimator_)


    w = csv.writer(open("eval/grid_results.csv", "w"))
    # Write the accuracy of each model in descending order
    for model in sorted(results_dict, key=results_dict.get, reverse = True):
        #print(model + "   " + str(results_dict[model]))
        w.writerow([model, results_dict[model]])
