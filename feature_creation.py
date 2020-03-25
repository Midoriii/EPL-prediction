import pandas as pd
import numpy as np

import zipfile
import glob
import json
import os
import sys



#Load previously saved sorted match info into a dataframe
def load_data():
    data1415 = pd.read_csv('data/season14-15/sorted_detailed_games_1415.csv', encoding='utf-8')
    data1516 = pd.read_csv('data/season15-16/sorted_detailed_games_1516.csv', encoding='utf-8')
    data1617 = pd.read_csv('data/season16-17/sorted_detailed_games_1617.csv', encoding='utf-8')
    data1718 = pd.read_csv('data/season17-18/sorted_detailed_games_1718.csv', encoding='utf-8')

    all_data = [data1415, data1516, data1617, data1718]

    #Gotta drop and reindex since the data contains a column full of 0
    for season in all_data:
        season.drop(season.columns[0], axis=1, inplace=True)
        season.reset_index(inplace=True, drop=True)

    #Get unique team names per season
    teams_per_season = []
    for season in all_data:
        teams_per_season.append(season.home_team_id.unique())


    return all_data, teams_per_season


#Create features for the whole season and last 5 matches from raw data
def create_features(data, team_ids):
    # Basis for extraction of matches before a certain match + selection of matches for all teams in the season

    data_with_features = []

    for season in data:

        #Empty dataframe which will hold the features
        season_dataframe = create_dataframe_of_features(season)

        #print(season_dataframe)

        #
        #Tady pribyde for loop pro kazdy match sezony
        #

        #Get all the matched played before the one we're building features for
        season = season[season['date_string'] < '2014-11-27 15:00:00']


        #We'll use the id of the match
        test_match = season.loc[season["match_id"] == 829587]
        #We'll get home team and away team ids
        id_of_test_team = test_match['home_team_id'].values[0]

        #And their matches so far
        test_team_matches = season[(season['home_team_id'] == id_of_test_team) | (season['away_team_id'] == id_of_test_team)]

        #Then their home & away matches
        home_matches = test_team_matches[test_team_matches['home_team_id'] == id_of_test_team]




        #And use it together with the season data to perform 'normalization' by dividing it by seasonal mean so far
        #Firstly for the att and def strength
        print(home_matches['full_time_score_home'].mean())
        print(home_matches['full_time_score_home'].mean() / season['full_time_score_away'].mean())
        
        #Afterwards for every other column

        #If the teams played less than 5 matches, just store the seasonal means into last5 means too

        #We'll need the matches of all the teams to perform means over last 5 played
        team_dict = dict()
        for team in team_ids[0]:
            games_of_a_team = season[(season['home_team_id'] == team) | (season['away_team_id'] == team)]
            team_dict[team] = games_of_a_team

        #print(team_dict)

        #lastly append the computed seasonal data
        data_with_features = data_with_features.append(season_dataframe)

    return data_with_features


#Dataframe creation to hold the desired features
def create_dataframe_of_features(season):
    #The dataframe which will hold the features
    season_dataframe = pd.DataFrame(columns=season.columns)
    #We'll use our own metric of strength
    season_dataframe = season_dataframe.assign(home_team_att_strength = "", home_team_def_strength = "", away_team_att_strength="",
                            away_team_def_strength = "")
    #Drop unimportant cols
    season_dataframe = season_dataframe.drop(columns=['date_string', 'half_time_score', 'half_time_score_away', 'half_time_score_home',
                                                      'full_time_score', 'full_time_score_away', 'full_time_score_home',])
    #Duplicate most of the cols for same metric during last 5 matches
    for col in season_dataframe.columns:
        if col not in ['match_id', 'result_home', 'home_team_id', 'away_team_id']:
            name = col + "_last5"
            season_dataframe[name] = ""

    return season_dataframe


#Main fuction for now
if __name__== "__main__":
    data, team_ids = load_data()

    data_with_features = create_features(data, team_ids)

    #Helper prints
    #print(data)
    #print(team_ids)

    #End of helper prints
