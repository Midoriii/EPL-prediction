import pandas as pd
import numpy as np

import os
import sys



#Helper function to get all/home/away matches of a team
def get_team_matches(season, team_id):
    #All their matches so far
    return season[(season['home_team_id'] == team_id) | (season['away_team_id'] == team_id)]


# Main fuction for now
if __name__== "__main__":

    # How many matches in a series ?
    if len(sys.argv) > 1:
        series_length = int(sys.argv[1])
    # If not specified, just do 5
    else:
        series_length = 3

    # Since we don't care about recent form when using RNNs, any of the normalized datasets is fine
    # Normalized should also work better with nets
    data_normalized = pd.DataFrame()
    data_normalized = data_normalized.append(pd.read_csv('data/season14-15/data_with_features_1415_norm_form3.csv', encoding='utf-8', index_col='match_id'))
    data_normalized = data_normalized.append(pd.read_csv('data/season15-16/data_with_features_1516_norm_form3.csv', encoding='utf-8', index_col='match_id'))
    data_normalized = data_normalized.append(pd.read_csv('data/season16-17/data_with_features_1617_norm_form3.csv', encoding='utf-8', index_col='match_id'))
    data_normalized = data_normalized.append(pd.read_csv('data/season17-18/data_with_features_1718_norm_form3.csv', encoding='utf-8', index_col='match_id'))

    # We want match_id as a column
    data_normalized.reset_index(inplace=True)
    # Remove columns containing recent form data
    data_normalized = data_normalized.loc[:,:'away_player_11']
    #print(data_normalized)

    # Here we'll store the outcome of a match that comes after a series
    labels = []
    # And here we'll store the series'
    Series_list = pd.DataFrame()

    #For every match, get the home team and away team, and get their last 4 matches to form a series
    for i in range(0,len(data_normalized)):
        #print(data_normalized.iloc[i]['match_id'])
        home_team = data_normalized.iloc[i]['home_team_id']
        away_team = data_normalized.iloc[i]['away_team_id']

        # Get the matches of the H and A teams, before the match in question
        home_team_matches_before = get_team_matches(data_normalized.iloc[0:i], home_team)
        away_team_matches_before = get_team_matches(data_normalized.iloc[0:i], away_team)

        # If there are enough matches played to form a series, form a series !
        if len(home_team_matches_before) >= series_length:
            # Append the last series_length matches as a series
            Series_list = Series_list.append(home_team_matches_before[- series_length:])
            # Append the result of this match as a label for prediction
            labels.append(data_normalized.iloc[i]['result_home'])
            #print(Series_list)
            #print(len(labels))
            #print(data_normalized.iloc[i]['match_id'])

        # Do the same for the away team
        if len(away_team_matches_before) >= series_length:
            Series_list = Series_list.append(away_team_matches_before[- series_length:])
            labels.append(data_normalized.iloc[i]['result_home'])

    # We don't need match ids for predictions and also the index values
    Series_list.drop(columns='match_id', inplace=True)
    Series_list.reset_index(drop=True, inplace=True)

    # Create data and labels as numpy arrays and reshape accordingly
    Y = np.array(labels)
    print(Y.shape)
    X = Series_list.to_numpy()
    X = X.reshape(int(len(Series_list)/series_length),series_length,59)
    print(X.shape)

    # Save for further usage
    np.save('rnn_data/labels_' + str(series_length), Y)
    np.save('rnn_data/data_' + str(series_length), X)
