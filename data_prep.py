import pandas as pd
import numpy as np

import zipfile
import glob
import json
import os

#Load json data and extract teams per season
def load_data():
    match_stats1415 = json.load(open('data/season14-15/season_match_stats.json','r'))
    match_stats1516 = json.load(open('data/season15-16/season_match_stats.json','r'))
    match_stats1617 = json.load(open('data/season16-17/season_match_stats.json','r'))
    match_stats1718 = json.load(open('data/season17-18/season_match_stats.json','r'))

    #Load simple match stats from each season
    matches1415 = pd.DataFrame.from_dict(match_stats1415, orient='index')
    matches1516 = pd.DataFrame.from_dict(match_stats1516, orient='index')
    matches1617 = pd.DataFrame.from_dict(match_stats1617, orient='index')
    matches1718 = pd.DataFrame.from_dict(match_stats1718, orient='index')

    all_seasons_matches = [matches1415, matches1516, matches1617, matches1718]

    #Get unique team names per season
    teams_per_season = []
    for matches in all_seasons_matches:
        teams_per_season.append(matches.home_team_name.unique())

    #Simple check
    for teams in teams_per_season:
        print(teams)

    return all_seasons_matches, teams_per_season


#Sort matches by datetime so we can generate GameWeeks
def sort_matches_by_date(all_seasons_matches):
    sorted_matches = []
    for matches in all_seasons_matches:
        #Convert datetime string to pandas datetime
        matches['date_string'] = pd.to_datetime(matches['date_string'], format='%d/%m/%Y %H:%M:%S')
        matches = matches.sort_values(by='date_string',ascending=True)
        matches = matches.reset_index()
        sorted_matches.append(matches)

    return sorted_matches


#Transform sorted matches into gameweeks
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


#Main fuction for now
if __name__== "__main__":
    all_matches_simple, teams_in_season = load_data()
    all_matches_simple = sort_matches_by_date(all_matches_simple)
    print(all_matches_simple)
    gameweeks_simple = make_gameweeks(all_matches_simple, teams_in_season)
    print(gameweeks_simple)
