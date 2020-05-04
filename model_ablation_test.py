import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.ensemble import ExtraTreesClassifier

import os
import sys



data = pd.DataFrame()
data = data.append(pd.read_csv('data/season14-15/data_with_features_1415_form5.csv', encoding='utf-8', index_col='match_id'))
data = data.append(pd.read_csv('data/season15-16/data_with_features_1516_form5.csv', encoding='utf-8', index_col='match_id'))
data = data.append(pd.read_csv('data/season16-17/data_with_features_1617_form5.csv', encoding='utf-8', index_col='match_id'))
data = data.append(pd.read_csv('data/season17-18/data_with_features_1718_form5.csv', encoding='utf-8', index_col='match_id'))

X_train = data[:320].append(data[370:690]).append(data[740:1060]).append(data[1110:1430])
X_test = data[320:370].append(data[690:740]).append(data[1060:1110]).append(data[1430:1480])

#Get the column names of features, we don't want the result here
col_names_features = list(data.columns)
col_names_features.remove('result_home')

#Second list of cols, from which we'll reduce features for ablation testing
col_names_features_ablation = list(data.columns)
col_names_features_ablation.remove('result_home')

# THIS WILL TOTALLY GET REFACTORED ONCE .. MAYBE

col_names_features_ablation.remove('home_team_id')
col_names_features_ablation.remove('away_team_id')

col_names_features_ablation.remove('home_shot_off_target')
col_names_features_ablation.remove('away_shot_off_target')
col_names_features_ablation.remove('away_ontarget_scoring_att')
col_names_features_ablation.remove('home_ontarget_scoring_att')
col_names_features_ablation.remove('home_won_corners')
col_names_features_ablation.remove('away_won_corners')
col_names_features_ablation.remove('home_team_att_strength')
col_names_features_ablation.remove('away_team_att_strength')
col_names_features_ablation.remove('home_team_def_strength')
col_names_features_ablation.remove('away_team_def_strength')
col_names_features_ablation.remove('away_blocked_scoring_att')
col_names_features_ablation.remove('home_blocked_scoring_att')
col_names_features_ablation.remove('home_wins')
col_names_features_ablation.remove('home_losses')
col_names_features_ablation.remove('away_wins')
col_names_features_ablation.remove('away_losses')
col_names_features_ablation.remove('home_possession_percentage')
col_names_features_ablation.remove('away_possession_percentage')
col_names_features_ablation.remove('home_total_pass')
col_names_features_ablation.remove('home_accurate_pass')
col_names_features_ablation.remove('away_total_pass')
col_names_features_ablation.remove('away_accurate_pass')
col_names_features_ablation.remove('away_team_rating')
col_names_features_ablation.remove('home_team_rating')

col_names_features_ablation.remove('home_shot_off_target_last5')
col_names_features_ablation.remove('away_shot_off_target_last5')
col_names_features_ablation.remove('away_ontarget_scoring_att_last5')
col_names_features_ablation.remove('home_ontarget_scoring_att_last5')
col_names_features_ablation.remove('home_won_corners_last5')
col_names_features_ablation.remove('away_won_corners_last5')
col_names_features_ablation.remove('home_team_att_strength_last5')
col_names_features_ablation.remove('away_team_att_strength_last5')
col_names_features_ablation.remove('home_team_def_strength_last5')
col_names_features_ablation.remove('away_team_def_strength_last5')
col_names_features_ablation.remove('away_blocked_scoring_att_last5')
col_names_features_ablation.remove('home_blocked_scoring_att_last5')
col_names_features_ablation.remove('home_wins_last5')
col_names_features_ablation.remove('home_losses_last5')
col_names_features_ablation.remove('away_wins_last5')
col_names_features_ablation.remove('away_losses_last5')
col_names_features_ablation.remove('home_possession_percentage_last5')
col_names_features_ablation.remove('away_possession_percentage_last5')
col_names_features_ablation.remove('home_total_pass_last5')
col_names_features_ablation.remove('home_accurate_pass_last5')
col_names_features_ablation.remove('away_total_pass_last5')
col_names_features_ablation.remove('away_accurate_pass_last5')
col_names_features_ablation.remove('away_team_rating_last5')
col_names_features_ablation.remove('home_team_rating_last5')

#This will be the Y, which we'll predict
col_names_outcome = 'result_home'

train_data = X_train[col_names_features]
train_data_abl = X_train[col_names_features_ablation]
train_labels = X_train[col_names_outcome]

print(train_data.shape)
print(train_data_abl.shape)

test_data = X_test[col_names_features]
test_data_abl = X_test[col_names_features_ablation]
test_labels = X_test[col_names_outcome]

model = ExtraTreesClassifier(n_estimators=600)

results_all_feats = []
results_ablated = []

for i in range(0,30):
    model.fit(train_data, train_labels)

    y_pred = model.predict(test_data)

    acc = accuracy_score(test_labels, y_pred)
    #print(acc)
    results_all_feats.append(acc)

# Let's see about the importance of each feature
importance_dict = {}
importance_dict_abl = {}

for i in range(0, len(col_names_features)):
    importance_dict[str(col_names_features[i])] = str(model.feature_importances_[i])

model = ExtraTreesClassifier(n_estimators=600)

for i in range(0,30):
    model.fit(train_data_abl, train_labels)

    y_pred = model.predict(test_data_abl)

    acc = accuracy_score(test_labels, y_pred)
    #print(acc)
    results_ablated.append(acc)

for i in range(0, len(col_names_features_ablation)):
    importance_dict_abl[str(col_names_features_ablation[i])] = str(model.feature_importances_[i])

for feat in sorted(importance_dict_abl, key=importance_dict_abl.get, reverse = True):
    print(feat + "   " + str(importance_dict_abl[feat]))

print("All feats: " + str(sum(results_all_feats) / 30))
print("Without features: " + str(sum(results_ablated) / 30))
