import pandas as pd
import numpy as np

import zipfile
import glob
import json
import os
import sys

#Load json data and extract teams per season
def load_data():
    match_stats1415 = json.load(open('data/season14-15/season_match_stats.json','r'))
    match_stats1516 = json.load(open('data/season15-16/season_match_stats.json','r'))
    match_stats1617 = json.load(open('data/season16-17/season_match_stats.json','r'))
    match_stats1718 = json.load(open('data/season17-18/season_match_stats.json','r'))

    season_stats1415 = json.load(open('data/season14-15/season_stats.json','r', encoding="utf8"))
    season_stats1516 = json.load(open('data/season15-16/season_stats.json','r', encoding="utf8"))
    season_stats1617 = json.load(open('data/season16-17/season_stats.json','r', encoding="utf8"))
    season_stats1718 = json.load(open('data/season17-18/season_stats.json','r', encoding="utf8"))

    #Load simple match stats from each season
    matches1415 = pd.DataFrame.from_dict(match_stats1415, orient='index')
    matches1516 = pd.DataFrame.from_dict(match_stats1516, orient='index')
    matches1617 = pd.DataFrame.from_dict(match_stats1617, orient='index')
    matches1718 = pd.DataFrame.from_dict(match_stats1718, orient='index')

    all_seasons_matches = [matches1415, matches1516, matches1617, matches1718]
    all_seasons_stats = [season_stats1415, season_stats1516, season_stats1617, season_stats1718]

    #Get unique team names per season .. probably not needed
    teams_per_season = []
    for matches in all_seasons_matches:
        teams_per_season.append(matches.home_team_name.unique())

    return all_seasons_matches, all_seasons_stats


#Sort matches by datetime, which will be needed when calculating aggregated stats about latest form,
#as well as whole season form
def sort_matches_by_date(all_seasons_matches):
    sorted_matches = []
    for matches in all_seasons_matches:
        #Convert datetime string to pandas datetime
        matches['date_string'] = pd.to_datetime(matches['date_string'], format='%d/%m/%Y %H:%M:%S')
        matches = matches.sort_values(by='date_string',ascending=True)
        #Reset and rename original index .. needed for sorting to remain
        matches = matches.reset_index()
        matches = matches.rename(columns={"index": "match_id"})
        sorted_matches.append(matches)

    return sorted_matches


#Transform sorted matches into gameweeks
#Is actually completely useless as it turns out
def make_gameweeks(matches, teams):
    #Helper index for seasons
    i = 0
    #Empty dataframe
    gameweek = pd.DataFrame()
    #An array of 10 match dataframes, actual gameweeks
    gameweeks = []
    #Iterate over all seasons
    for season in matches:
        #We'll be removing matches already taken care of
        while not season.empty:
            #Keep a list of teams that need to be paired in gameweek
            team_list = teams[i]
            #Gotta convert it to list
            team_list = team_list.tolist()

            for idx, row in season.iterrows():
                #All teams are paired, go again
                if len(team_list) == 0:
                    #Record the gameweek
                    gameweeks.append(gameweek)
                    #Clean slate
                    gameweek = pd.DataFrame()
                    #Onto the next gameweek
                    break
                #We found a pair to match
                if row['home_team_name'] in team_list and row['away_team_name'] in team_list:
                    #Append the row into gameweeks
                    gameweek = gameweek.append(row)
                    #Remove both teams from the list of unmatched teams
                    team_list.remove(row['home_team_name'])
                    team_list.remove(row['away_team_name'])
                    #Remove from matches
                    season = season.drop(idx)
                    #print(row)
        i += 1
    return gameweeks


#Add detailed info about home and away teams
#The final data containing aggregated stats will be built using these
def make_detailed_matches(matches_simple, matches_detailed):
    matches_combined = []
    #For all seasons
    for i in range(0,4):
        #Empty dataframe to append to
        season = pd.DataFrame()
        #For every match in the season
        for idx, row in matches_simple[i].iterrows():
            #Empty dataframe for every match, which will be filled
            single_match = pd.DataFrame(columns=["match_id","date_string","home_team_id","away_team_id",
                                                 "half_time_score","full_time_score","home_team_rating",
                                                 "home_accurate_pass", "home_total_pass", "home_blocked_scoring_att",
                                                 "home_ontarget_scoring_att", "home_shot_off_target", "home_won_corners",
                                                 "home_penalty_save", "home_total_tackle", "home_fk_foul_lost",
                                                 "home_total_throws", "home_possession_percentage",
                                                 "away_team_rating", "away_accurate_pass", "away_total_pass",
                                                 "away_blocked_scoring_att", "away_ontarget_scoring_att",
                                                 "away_shot_off_target", "away_won_corners", "away_penalty_save",
                                                 "away_total_tackle", "away_fk_foul_lost",
                                                 "away_total_throws", "away_possession_percentage"])
            #Create blank row to fill data in
            single_match.loc[0] = 0
            #Columns we want from the simple match stats
            simple_row_columns = ["match_id","date_string","home_team_id","away_team_id",
                                  "half_time_score","full_time_score"]
            #Columns we want from the detailed match stats
            team_stats_columns = ["team_rating", "accurate_pass", "total_pass", "blocked_scoring_att",
                                  "ontarget_scoring_att", "shot_off_target", "won_corners", "penalty_save",
                                  "total_tackle", "fk_foul_lost", "total_throws", "possession_percentage"]

            home_team = row['home_team_id']
            away_team = row['away_team_id']

            #Fill dataframe with desired values from simple stats
            for key in simple_row_columns:
                single_match.loc[0,key] = row[key]
            #Fill dataframe with desired values for home team from detailed stats
            for key in team_stats_columns:
                #We need to check if key exists
                if key in matches_detailed[i][row['match_id']][home_team]['team_details']:
                    single_match.loc[0,"home_" + key] = matches_detailed[i][row['match_id']][home_team]['team_details'][key]
                elif key in matches_detailed[i][row['match_id']][home_team]['aggregate_stats']:
                    single_match.loc[0,"home_" + key] = matches_detailed[i][row['match_id']][home_team]['aggregate_stats'][key]
                #If the key doesn't exist, just fill 0
                else:
                    single_match.loc[0,"home_" + key] = 0
             #Fill dataframe with desired values for away team from detailed stats
            for key in team_stats_columns:
                if key in matches_detailed[i][row['match_id']][away_team]['team_details']:
                    single_match.loc[0,"away_" + key] = matches_detailed[i][row['match_id']][away_team]['team_details'][key]
                elif key in matches_detailed[i][row['match_id']][away_team]['aggregate_stats']:
                    single_match.loc[0,"away_" + key] = matches_detailed[i][row['match_id']][away_team]['aggregate_stats'][key]
                else:
                    single_match.loc[0,"away_" + key] = 0

            single_match.fillna(0)
            #We want numbers as ints, not objects
            for atrib in team_stats_columns:
                if atrib not in ['possession_percentage', 'team_rating']:
                    single_match['home_' + atrib] = single_match['home_' + atrib].astype(np.int64)
                    single_match['away_' + atrib] = single_match['away_' + atrib].astype(np.int64)

            single_match['home_possession_percentage'] = single_match['home_possession_percentage'].astype(np.float64)
            single_match['away_possession_percentage'] = single_match['away_possession_percentage'].astype(np.float64)
            single_match['home_team_rating'] = single_match['home_team_rating'].astype(np.float64)
            single_match['away_team_rating'] = single_match['away_team_rating'].astype(np.float64)

            single_match['date_string'] = pd.to_datetime(single_match['date_string'])

            season = season.append(single_match)

        matches_combined.append(season)

    return matches_combined


#Divide scores into goals for home and away sides and add the home team result - W/L/D
def divide_score_and_add_result(matches_combined):
    for i in range (0,4):
        #Divide the scores
        matches_combined[i] = divideScore(matches_combined[i], 'half_time_score','half_time_score_home',
                                          'half_time_score_away')
        matches_combined[i] = divideScore(matches_combined[i], 'full_time_score', 'full_time_score_home',
                                          'full_time_score_away')
        #Retype for comparison
        matches_combined[i] = matches_combined[i].astype({'half_time_score_home': 'int8',
                                                'half_time_score_away': 'int8',
                                                'full_time_score_home': 'int8',
                                                'full_time_score_away': 'int8'})
        #Transform results into representation, D/L/W in order, we'll be predicting these
        matches_combined[i].loc[matches_combined[i]['full_time_score_home'] == matches_combined[i]['full_time_score_away'],
                                'result_home'] = 0
        matches_combined[i].loc[matches_combined[i]['full_time_score_home'] < matches_combined[i]['full_time_score_away'],
                                'result_home'] = 2
        matches_combined[i].loc[matches_combined[i]['full_time_score_home'] > matches_combined[i]['full_time_score_away'],
                                'result_home'] = 1
    return matches_combined


#Helper function to divide score and store it back into the dataframe
def divideScore(df, scoreName, name1, name2):
    df2 = df[scoreName].str.split(':',expand=True)
    df2.rename(columns={0: name1, 1: name2}, inplace=True)
    df2 = pd.concat([df, df2], axis=1)

    return df2


#Main fuction for now
if __name__== "__main__":
    all_matches_simple, all_matches_detailed = load_data()
    all_matches_simple = sort_matches_by_date(all_matches_simple)
    all_matches_merged_sorted = make_detailed_matches(all_matches_simple, all_matches_detailed)
    all_matches_merged_sorted = divide_score_and_add_result(all_matches_merged_sorted)

    #Save it all for future use
    # Bug: the indices get lost somewhere along the way and become 0 resulting in Unnamed col full of 0
    all_matches_merged_sorted[0].to_csv('data/season14-15/sorted_detailed_games_1415.csv', encoding='utf-8')
    all_matches_merged_sorted[1].to_csv('data/season15-16/sorted_detailed_games_1516.csv', encoding='utf-8')
    all_matches_merged_sorted[2].to_csv('data/season16-17/sorted_detailed_games_1617.csv', encoding='utf-8')
    all_matches_merged_sorted[3].to_csv('data/season17-18/sorted_detailed_games_1718.csv', encoding='utf-8')

    #A Bunch of helper prints
    #print(all_matches_simple)
    #print(all_matches_merged_sorted)
    #print(all_matches_detailed)
    #print(all_matches_simple[0]['match_id'])
    #print(all_matches_detailed[0]['829513']['13'])

    #Helper prints over
