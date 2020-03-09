import pandas as pd
import numpy as np

import zipfile
import glob
import json
import os

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

for matches in all_seasons_matches:
    print(matches.shape)

#Get unique team names per season
teams_per_season = []
for matches in all_seasons_matches:
    teams_per_season.append(matches.home_team_name.unique())
    
#Simple check
for teams in teams_per_season:
    print(teams)

#Sort matches by datetime so we can generate GameWeeks
for matches in all_seasons_matches:
    #Convert datetime string to pandas datetime
    matches['date_string'] = pd.to_datetime(matches['date_string'], infer_datetime_format=True)
    matches = matches.sort_values(by='date_string',ascending=True)
    print(matches)
