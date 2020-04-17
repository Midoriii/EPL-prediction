** results.csv ** contains the sorted models by their achieved accuracy over 100 training and testing trials on randomly selected data splits (85/15 train/test)

** end_of_season_results.csv ** contains the same but the testing set is made of the last 50 matches of each season


The difference is that in the first case, the randomness of the split might include matches from the beginning of the season, for which my method based on 
seasonal and recent performance might not produce the best results, since obviously there's not enough good info about such matches.
This somewhat reflects on the results, as end of the season predictions score a bit higher. 

It can also be seen that the end of the season predictions favour recent form span over 5 to 8 matches, while randomly selected matches lean towards more recent form of 3 matches.



The 'Normalized' and 'Reduced' descriptions mean that the features used were normalized or their amount reduced using SelectKBest respectively.