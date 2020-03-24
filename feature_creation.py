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



#Main fuction for now
if __name__== "__main__":
    data, team_ids = load_data()

    for season in data:
        season = season[season['date_string'] < '2014-09-27 15:00:00']
        print(season)

    #Helper prints
    #print(data)
    #print(team_ids)

    #End of helper prints
