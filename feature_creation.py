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

        #For every match in the season
        for idx, row in season.iterrows():
            #Empty dataframe which will be filled with single match data and appended to season_dataframe
            match_dataframe = create_dataframe_of_features(season)
            match_dataframe.loc[0] = 0
            match_dataframe.loc[0]['match_id'] = row['match_id']
            match_dataframe.loc[0]['home_team_id'] = row['home_team_id']
            match_dataframe.loc[0]['away_team_id'] = row['away_team_id']
            match_dataframe.loc[0]['result_home'] = row['result_home']

            #We'll use the id of the match
            match_id = row['match_id']

            #Get all the matches played before the one we're building features for
            matches_before = season[season['date_string'] < row['date_string']]

            #We'll get home team and away team ids
            id_of_H_team = row['home_team_id']
            id_of_A_team = row['away_team_id']

            #Get all their matches so far + home / away
            teamH_all_matches, teamH_home_matches, teamH_away_matches = get_team_matches(matches_before, id_of_H_team)
            teamA_all_matches, teamA_home_matches, teamA_away_matches = get_team_matches(matches_before, id_of_A_team)

            #We'll need to skip the first week since we have no prior data
            if(len(teamA_all_matches) == 0 or len(teamH_all_matches) == 0):
                #Skip the first matches of a season .. will probably be for the best
                continue

            #And use it together with the season data to perform 'normalization' by dividing it by seasonal mean so far
            #Firstly for the att and def strength
            match_dataframe.loc[0]['home_team_att_strength'] = teamH_home_matches['full_time_score_home'].mean() / matches_before['full_time_score_home'].mean()
            match_dataframe.loc[0]['home_team_def_strength'] = teamH_home_matches['full_time_score_away'].mean() / matches_before['full_time_score_away'].mean()
            match_dataframe.loc[0]['away_team_att_strength'] = teamA_away_matches['full_time_score_away'].mean() / matches_before['full_time_score_away'].mean()
            match_dataframe.loc[0]['away_team_def_strength'] = teamA_away_matches['full_time_score_home'].mean() / matches_before['full_time_score_home'].mean()

            #Get the column names
            col_names_home = list(match_dataframe.columns[3:15].values)
            col_names_away = list(match_dataframe.columns[15:28].values)
            #Ugly but whatever
            col_names_away.remove('result_home')

            #Afterwards for every desired column compute the mean for the team during the whole season
            for i in range(len(col_names_away)):
                #Get the mean of a stat for the current home team
                home_mean_home_matches = teamH_home_matches[col_names_home[i]].mean()
                home_mean_away_matches = teamH_away_matches[col_names_away[i]].mean()
                home_mean = (home_mean_away_matches + home_mean_home_matches) / 2.0
                match_dataframe.loc[0][col_names_home[i]] = home_mean

                #Get the mean of a stat for the current away team
                away_mean_home_matches = teamA_home_matches[col_names_home[i]].mean()
                away_mean_away_matches = teamA_away_matches[col_names_away[i]].mean()
                away_mean = (away_mean_away_matches + away_mean_home_matches) / 2.0
                match_dataframe.loc[0][col_names_away[i]] = away_mean


            #If the teams played less than 5 matches, just store the seasonal means into last5 means too
            if(len(teamA_all_matches) < 5):
                #equals
                print('eh')
            else:
                #compute
                print('ho')
            #Equally for the home team
            if(len(teamH_all_matches) < 5):
                #equals
                print('eh')
            else:
                #compute
                print('ho')

            #We'll need the matches of all the teams to perform means over last 5 played
            team_dict = dict()
            for team in team_ids[0]:
                games_of_a_team = season[(season['home_team_id'] == team) | (season['away_team_id'] == team)]
                team_dict[team] = games_of_a_team

            #print(team_dict)

            print(match_dataframe)

            #Fill NaNs and 0 in the beginning as 1, which should be average
            match_dataframe.fillna(value=1, inplace=True)
            match_dataframe.replace(0, 1, inplace=True)

            #Append constructed match to the season dataframe
            season_dataframe = season_dataframe.append(match_dataframe)

        #lastly append the computed seasonal data
        data_with_features.append(season_dataframe)

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


#Helper function to get all/home/away matches of a team
def get_team_matches(season, team_id):
    #All their matches so far
    all_matches = season[(season['home_team_id'] == team_id) | (season['away_team_id'] == team_id)]

    #Then their home & away matches
    home_matches = all_matches[all_matches['home_team_id'] == team_id]
    away_matches = all_matches[all_matches['away_team_id'] == team_id]

    return all_matches, home_matches, away_matches



#Main fuction for now
if __name__== "__main__":
    data, team_ids = load_data()

    data_with_features = create_features(data, team_ids)

    data_with_features[0].to_csv('data/season14-15/data_with_features_1415.csv', encoding='utf-8')
    data_with_features[1].to_csv('data/season15-16/data_with_features_1516.csv', encoding='utf-8')
    data_with_features[2].to_csv('data/season16-17/data_with_features_1617.csv', encoding='utf-8')
    data_with_features[3].to_csv('data/season17-18/data_with_features_1718.csv', encoding='utf-8')

    #Helper prints
    #print(data)
    #print(team_ids)

    #End of helper prints
