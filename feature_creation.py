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


#Create features for the whole season and last N matches from raw data
def create_features(data, team_ids, recent_form):
    # Basis for extraction of matches before a certain match + selection of matches for all teams in the season

    data_with_features = []

    for season in data:

        #Empty dataframe which will hold the features
        season_dataframe = create_dataframe_of_features(season, recent_form)

        #print(season_dataframe)

        #For every match in the season
        for idx, row in season.iterrows():
            #Empty dataframe which will be filled with single match data and appended to season_dataframe
            match_dataframe = create_dataframe_of_features(season, recent_form)
            match_dataframe.loc[0] = 0
            match_dataframe.loc[0]['match_id'] = row['match_id']
            match_dataframe.loc[0]['home_team_id'] = row['home_team_id']
            match_dataframe.loc[0]['away_team_id'] = row['away_team_id']

            #Get the column names
            col_names_home = list(match_dataframe.columns[3:15].values)
            col_names_away = list(match_dataframe.columns[15:28].values)
            #Isn't needed there
            col_names_away.remove('result_home')

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

            #We define relative home & away strengths as team means divided by seasonal means
            #First few weeks can contain NaNs so we replace those by average represented as 1.0
            home_att_str = teamH_home_matches['full_time_score_home'].mean() / matches_before['full_time_score_home'].mean()
            match_dataframe.loc[0]['home_team_att_strength'] = (1.0 if np.isnan(home_att_str) else home_att_str)
            home_def_str = teamH_home_matches['full_time_score_away'].mean() / matches_before['full_time_score_away'].mean()
            match_dataframe.loc[0]['home_team_def_strength'] = (1.0 if np.isnan(home_def_str) else home_def_str)
            away_att_str = teamA_away_matches['full_time_score_away'].mean() / matches_before['full_time_score_away'].mean()
            match_dataframe.loc[0]['away_team_att_strength'] = (1.0 if np.isnan(away_att_str) else away_att_str)
            away_def_str = teamA_away_matches['full_time_score_home'].mean() / matches_before['full_time_score_home'].mean()
            match_dataframe.loc[0]['away_team_def_strength'] = (1.0 if np.isnan(away_def_str) else away_def_str)

            #Compute the features for the Home team and Away team afterwards
            compute_features(col_names_away, col_names_home, teamH_home_matches, teamH_away_matches, True, match_dataframe)
            compute_features(col_names_away, col_names_home, teamA_home_matches, teamA_away_matches, False, match_dataframe)

            #If the teams played less than N matches, just store the seasonal means into lastN means too
            if(len(teamA_all_matches) < recent_form):
                for col in col_names_away:
                    match_dataframe.loc[0][col + "_last" + str(recent_form)] = match_dataframe[col].values[0]
                match_dataframe.loc[0]["away_team_att_strength_last" + str(recent_form)] = match_dataframe["away_team_att_strength"].values[0]
                match_dataframe.loc[0]["away_team_def_strength_last" + str(recent_form)] = match_dataframe["away_team_def_strength"].values[0]

            else:
                #Get combined last N matches
                lastN_matches_of_teams = get_lastN_team_matches(matches_before, team_ids, recent_form)
                #Get matches of the current away team
                teamA_lastN_all_matches, teamA_lastN_home_matches, teamA_lastN_away_matches = get_team_matches(lastN_matches_of_teams, id_of_A_team)
                #Compute the features of the last N matches
                compute_features(col_names_away, col_names_home, teamA_lastN_home_matches, teamA_lastN_away_matches, False, match_dataframe, "_last" + str(recent_form))
                #Need to do relative strengths too
                away_att_str = teamA_lastN_away_matches['full_time_score_away'].mean() / lastN_matches_of_teams['full_time_score_away'].mean()
                match_dataframe.loc[0]['away_team_att_strength_last' + str(recent_form)] = (1.0 if np.isnan(away_att_str) else away_att_str)
                away_def_str = teamA_lastN_away_matches['full_time_score_home'].mean() / lastN_matches_of_teams['full_time_score_home'].mean()
                match_dataframe.loc[0]['away_team_def_strength_last' + str(recent_form)] = (1.0 if np.isnan(away_def_str) else away_def_str)


            #Equally for the home team
            if(len(teamH_all_matches) < N):
                for col in col_names_home:
                    match_dataframe.loc[0][col + "_last" + str(recent_form)] = match_dataframe[col].values[0]
                match_dataframe.loc[0]["home_team_att_strength_last" + str(recent_form)] = match_dataframe["home_team_att_strength"].values[0]
                match_dataframe.loc[0]["home_team_def_strength_last" + str(recent_form)] = match_dataframe["home_team_def_strength"].values[0]
            else:
                lastN_matches_of_teams = get_lastN_team_matches(matches_before, team_ids, recent_form)
                teamH_lastN_all_matches, teamH_lastN_home_matches, teamH_lastN_away_matches = get_team_matches(lastN_matches_of_teams, id_of_H_team)
                compute_features(col_names_away, col_names_home, teamH_lastN_home_matches, teamH_lastN_away_matches, True, match_dataframe, "_last" + str(recent_form))

                home_att_str = teamH_lastN_home_matches['full_time_score_home'].mean() / lastN_matches_of_teams['full_time_score_home'].mean()
                match_dataframe.loc[0]['home_team_att_strength_last' + str(recent_form)] = (1.0 if np.isnan(home_att_str) else home_att_str)
                home_def_str = teamH_lastN_home_matches['full_time_score_away'].mean() / lastN_matches_of_teams['full_time_score_away'].mean()
                match_dataframe.loc[0]['home_team_def_strength_last' + str(recent_form)] = (1.0 if np.isnan(home_def_str) else home_def_str)


            #print(match_dataframe)

            #Fill NaNs if some appear
            match_dataframe.fillna(value=-1, inplace=True)

            #Append constructed match to the season dataframe
            season_dataframe = season_dataframe.append(match_dataframe)

        #lastly append the computed seasonal data
        data_with_features.append(season_dataframe)

    return data_with_features


#Dataframe creation to hold the desired features
def create_dataframe_of_features(season, recent_form):
    #The dataframe which will hold the features
    season_dataframe = pd.DataFrame(columns=season.columns)
    #We'll use our own metric of strength
    season_dataframe = season_dataframe.assign(home_team_att_strength = "", home_team_def_strength = "", away_team_att_strength="",
                            away_team_def_strength = "", home_wins = "", away_wins = "", home_draws = "", away_draws ="",
                            home_losses = "", away_losses = "")
    #Drop unimportant cols
    season_dataframe = season_dataframe.drop(columns=['date_string', 'half_time_score', 'half_time_score_away', 'half_time_score_home',
                                                      'full_time_score', 'full_time_score_away', 'full_time_score_home',])
    #Duplicate most of the cols for same metric during last N matches
    for col in season_dataframe.columns:
        if col not in ['match_id', 'result_home', 'home_team_id', 'away_team_id']:
            name = col + "_last" + str(recent_form)
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


#For every desired column compute the mean for the team during the whole season
#Suffix is there to make it reusable for last N matches
def compute_features(col_names_away, col_names_home, home_matches, away_matches, home_team, match_dataframe, suffix=""):
    for i in range(len(col_names_home)):
        #Get the mean of a stat for the current home team
        mean_home_matches = home_matches[col_names_home[i]].mean()
        mean_away_matches = away_matches[col_names_away[i]].mean()
        #This here is to 'fix' the second gameweek mainly, where away or home matches may be missing
        #and outputting NaN
        if np.isnan(mean_away_matches):
            mean_away_matches = mean_home_matches
        if np.isnan(mean_home_matches):
            mean_home_matches = mean_away_matches

        mean = (mean_away_matches + mean_home_matches) / 2.0
        #Are we computing this for the home or away team ?
        if home_team:
            match_dataframe.loc[0][col_names_home[i]+suffix] = mean
        else:
            match_dataframe.loc[0][col_names_away[i]+suffix] = mean

    # Compute Wins Draws and Losses too
    wins = 0
    losses = 0
    draws = 0

    # Scan through the home matches and increase appropriate result count
    for idx, row in home_matches.iterrows():
        if row['result_home'] == 1:
            wins += 1
        elif row['result_home'] == -1:
            losses += 1
        else:
            draws += 1

    # Same for away matches
    for idx, row in away_matches.iterrows():
        if row['result_home'] == 1:
            losses += 1
        elif row['result_home'] == -1:
            wins += 1
        else:
            draws += 1

    # Write the results to the dataframe
    if home_team:
        match_dataframe.loc[0]['home_wins'+suffix] = wins
        match_dataframe.loc[0]['home_draws'+suffix] = draws
        match_dataframe.loc[0]['home_losses'+suffix] = losses
    else:
        match_dataframe.loc[0]['away_wins'+suffix] = wins
        match_dataframe.loc[0]['away_draws'+suffix] = draws
        match_dataframe.loc[0]['away_losses'+suffix] = losses



#Get last N matches of all teams, used in computing features over last 5 games to emphasize recent form
def get_lastN_team_matches(matches_before, team_ids, recent_form):
    #We'll need the matches of all the teams to perform means over last 5 played
    #Empty dataframe to append to
    lastN_games_of_all_teams = pd.DataFrame()
    team_dict = dict()
    #Get all games for each team and store in a dictionary where key = team id
    for team in team_ids[0]:
        games_of_a_team = matches_before[(matches_before['home_team_id'] == team) | (matches_before['away_team_id'] == team)]
        team_dict[team] = games_of_a_team

    for key in team_dict:
        #If a team has played less than N matches, append them all .. might actually not even be needed here
        if len(team_dict[key].index) < recent_form:
            lastN_games_of_all_teams = lastN_games_of_all_teams.append(team_dict[key])
        #Otherwise append exactly last N
        else:
            lastN_games_of_all_teams = lastN_games_of_all_teams.append(team_dict[key].tail(recent_form))

    #Drop duplicated matches
    lastN_games_of_all_teams.drop_duplicates(subset='match_id', inplace=True)

    #print(lastN_games_of_all_teams)

    #Return dataframe containing last ~N matches of each team
    return lastN_games_of_all_teams


#Normalize data to see if it improves performance
def normalize_matchframe(df):
    result = df.copy()
    #Get the feature names
    features = df.columns
    for feature in features:
        #We don't wanna normalize these
        if feature not in ['match_id', 'result_home', 'home_team_id', 'away_team_id']:
            max_value = df[feature].max()
            min_value = df[feature].min()
            result[feature] = (df[feature] - min_value) / (max_value - min_value)
    return result


#Main fuction for now
if __name__== "__main__":
    data, team_ids = load_data()

    # We'll be using features describing recent form, namely for the last 3/5/8 matches
    # to see if there's a difference and what works the best
    for N in [3, 5, 8]:

        data_with_features = create_features(data, team_ids, N)

        data_with_features[0].to_csv('data/season14-15/data_with_features_1415_form' + str(N) + '.csv', encoding='utf-8', index=False)
        data_with_features[1].to_csv('data/season15-16/data_with_features_1516_form' + str(N) + '.csv', encoding='utf-8', index=False)
        data_with_features[2].to_csv('data/season16-17/data_with_features_1617_form' + str(N) + '.csv', encoding='utf-8', index=False)
        data_with_features[3].to_csv('data/season17-18/data_with_features_1718_form' + str(N) + '.csv', encoding='utf-8', index=False)

        normalize_matchframe(data_with_features[0]).to_csv('data/season14-15/data_with_features_1415_norm_form' + str(N) + '.csv', encoding='utf-8', index=False)
        normalize_matchframe(data_with_features[1]).to_csv('data/season15-16/data_with_features_1516_norm_form' + str(N) + '.csv', encoding='utf-8', index=False)
        normalize_matchframe(data_with_features[2]).to_csv('data/season16-17/data_with_features_1617_norm_form' + str(N) + '.csv', encoding='utf-8', index=False)
        normalize_matchframe(data_with_features[3]).to_csv('data/season17-18/data_with_features_1718_norm_form' + str(N) + '.csv', encoding='utf-8', index=False)


    #Helper prints
    #print(data)
    #print(team_ids)

    #End of helper prints
