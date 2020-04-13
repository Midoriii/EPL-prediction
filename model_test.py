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

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC



# Function to fit desired classifier on given data and store the performance on training set in a
# dictionary containing results, by returning the acc
def evaluate_model(X_train, Y_train, X_test, Y_test, model):
    # Train classifier
    model.fit(
        X_train,
        Y_train
    )
    # Predict on test data
    y_pred = model.predict(X_test)

    # Return the performance/accuracy in %
    return accuracy_score(Y_test, y_pred)


# Get the column names that are most important according to selectKbest features
def get_reduced_columns(N, data, labels):
    selector = SelectKBest(chi2, k=N)
    #Fit the selector on data
    selector.fit(data, labels)
    #Which features are selected ?
    mask = selector.get_support(indices=True)
    #Return names of the most important cols
    return data.columns[mask]



# Main fuction for now
if __name__== "__main__":

    # Dictionary to store results corresponding to model name
    results_dict = defaultdict(lambda: 0)

    # How many testing iterations
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    # If not specified, just do 5
    else:
        iterations = 5


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

        for i in range (0,iterations):

            print(i)

            # Split dataset in training and test datasets
            X_train, X_test = train_test_split(data, test_size=0.15, random_state=int(time.time()))
            # Split normalized dataset in training and test datasets
            Y_train, Y_test = train_test_split(data_normalized, test_size=0.15, random_state=int(time.time()))

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
            reduced_features_20 = get_reduced_columns(20, train_data, train_labels)
            #Do again for normalized data
            reduced_features_normalized_20 = get_reduced_columns(20, train_data_norm, train_labels_norm)

            reduced_features_30 = get_reduced_columns(30, train_data, train_labels)
            reduced_features_normalized_30 = get_reduced_columns(30, train_data_norm, train_labels_norm)

            reduced_features_40 = get_reduced_columns(40, train_data, train_labels)
            reduced_features_normalized_40 = get_reduced_columns(40, train_data_norm, train_labels_norm)

            reduced_features_50 = get_reduced_columns(50, train_data, train_labels)
            reduced_features_normalized_50 = get_reduced_columns(50, train_data_norm, train_labels_norm)

            reduced_features_60 = get_reduced_columns(60, train_data, train_labels)
            reduced_features_normalized_60 = get_reduced_columns(60, train_data_norm, train_labels_norm)

            #Create models
            gnb = GaussianNB()
            mnb = MultinomialNB()

            knn_15 = KNeighborsClassifier(n_neighbors=15)
            knn_25 = KNeighborsClassifier(n_neighbors=25)
            knn_35 = KNeighborsClassifier(n_neighbors=35)
            knn_50 = KNeighborsClassifier(n_neighbors=50)
            knn_70 = KNeighborsClassifier(n_neighbors=70)
            knn_100 = KNeighborsClassifier(n_neighbors=100)
            knn_120 = KNeighborsClassifier(n_neighbors=120)

            svm = OneVsRestClassifier(SVC(C=1.5, gamma='auto'))
            svm_2 = OneVsRestClassifier(SVC(C=10, gamma='auto'))

            forest = RandomForestClassifier(n_estimators=50)
            forest_2 = RandomForestClassifier(n_estimators=100)
            forest_3 = RandomForestClassifier(n_estimators=200)
            forest_4 = RandomForestClassifier(n_estimators=400)
            forest_5 = RandomForestClassifier(n_estimators=600)
            forest_6 = RandomForestClassifier(n_estimators=1000)

            extra_forest = ExtraTreesClassifier(n_estimators=100)
            extra_forest_2 = ExtraTreesClassifier(n_estimators=200)
            extra_forest_3 = ExtraTreesClassifier(n_estimators=400)
            extra_forest_4 = ExtraTreesClassifier(n_estimators=600)
            extra_forest_5 = ExtraTreesClassifier(n_estimators=1000)

            ada = AdaBoostClassifier(n_estimators=50)
            ada_2 = AdaBoostClassifier(n_estimators=100)
            ada_3 = AdaBoostClassifier(n_estimators=200)


            models = [gnb, mnb, knn_15, knn_25, knn_35, knn_50, knn_70, knn_100, knn_120, svm, svm_2,
                      forest, forest_2, forest_3, forest_4, forest_5, forest_6, ada, ada_2, ada_3,
                      extra_forest, extra_forest_2, extra_forest_3, extra_forest_4, extra_forest_5]
            names_of_models = ["Gaussian NB", "Multinomial NB", "KNN-15", "KNN-25", "KNN-35", "KNN-50", "KNN-70",
                               "KNN-100", "KNN-120", "SVM", "SVM_2", "Random Forest-50", "Random Forest-100",
                               "Random Forest-200", "Random Forest-400", "Random Forest-600",
                               "Random Forest-1000","AdaBoost-50", "AdaBoost-100", "AdaBoost-200",
                               "Extreme Forest-100", "Extreme Forest-200", "Extreme Forest-400",
                               "Extreme Forest-600", "Extreme Forest-1000"]

            # Things to zip together, to make the final loop more readable
            features = [reduced_features_20, reduced_features_30, reduced_features_40, reduced_features_50, reduced_features_60]
            features_normalized = [reduced_features_normalized_20, reduced_features_normalized_30, reduced_features_normalized_40, reduced_features_normalized_50, reduced_features_normalized_60]

            descriptions = ["  form: " + str(N) + "   Reduced-20",
                            "  form: " + str(N) + "   Reduced-30",
                            "  form: " + str(N) + "   Reduced-40",
                            "  form: " + str(N) + "   Reduced-50",
                            "  form: " + str(N) + "   Reduced-60"]
            descriptions_normalized = ["  form: " + str(N) + "   Normalized   Reduced-20",
                            "  form: " + str(N) + "   Normalized   Reduced-30",
                            "  form: " + str(N) + "   Normalized   Reduced-40",
                            "  form: " + str(N) + "   Normalized   Reduced-50",
                            "  form: " + str(N) + "   Normalized   Reduced-60"]

            # Run predictions on data and normalized data for every model
            for model, name in zip(models, names_of_models):
                # Compute accuraccy for raw data
                results_dict[name + "  form: " + str(N)] += evaluate_model(train_data,
                                                                          train_labels,
                                                                          test_data,
                                                                          test_labels,
                                                                          model)
                # And for normalized data
                results_dict[name + "  form: " + str(N) + "   Normalized"] += evaluate_model(train_data_norm,
                                                                          train_labels_norm,
                                                                          test_data_norm,
                                                                          test_labels_norm,
                                                                          model)

                # Now do the same for reduced features
                for feats, feats_n, desc, desc_n in zip(features, features_normalized, descriptions, descriptions_normalized):
                    results_dict[name + desc] += evaluate_model(train_data[feats],
                                                                              train_labels,
                                                                              test_data[feats],
                                                                              test_labels,
                                                                              model)

                    results_dict[name + desc_n] += evaluate_model(train_data_norm[feats_n],
                                                                              train_labels_norm,
                                                                              test_data_norm[feats_n],
                                                                              test_labels_norm,
                                                                              model)


    # Average the results
    for key in results_dict.keys():
        results_dict[key] /= iterations

    w = csv.writer(open("eval/res.csv", "w"))
    # Write the accuracy of each model in descending order
    for model in sorted(results_dict, key=results_dict.get, reverse = True):
        #print(model + "   " + str(results_dict[model]))
        w.writerow([model, results_dict[model]])
