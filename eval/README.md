**results.csv** contains the sorted models by their achieved accuracy over 100 training and testing trials on randomly selected data splits (85/15 train/test)

**end_of_season_results.csv** contains the same but the testing set is made of the last 50 matches of each season

**detailed_results.csv** contains the precision / recall / f1-score data about each model

The difference is that in the first case, the randomness of the split might include matches from the beginning of the season, for which my method based on
seasonal and recent performance might not produce the best results, since obviously there's not enough good info about such matches.
This somewhat reflects on the results, as end of the season predictions score a bit higher.

It can also be seen that the end of the season predictions favour recent form span over 5 to 8 matches, while randomly selected matches lean towards more recent form of 3 matches.

From the detailed results, one can see that most models are terrible at predicting draws, which
is something even human experts have trouble with. The best models, which are forests, perform well at
predicting home side victories (especially recall - few false negatives) and reasonable well at away victories (both home and away score about 0.55 in precision, while away victories have substantially lower
recall - 0.8 vs 0.48).

Their **final** variants mean that those were used for the final report.


**rnn_results.txt** contains the loss and accuracy of all three RNN models (Shallow, Stacked, Bidirectional, in that order) across datasets of series lengths of 2/3/4/5/8.

As far as RNNs are concerned, the best length of the input series is 2 or 3 matches, which is kind of expected, since football matches are played weekly and most teams rarely hold on to their form for a longer period of time than 1 month (4 matches). There isn't much difference between the models, Stacked one has a better loss score by a slight margin, but accuracy wise they're equal. This approach might work, but most likely requires more data than 4 seasons.


**grid_results.csv** contains the best results of differently parameterized forests, which showed to be the best classifiers, through grid search

**grid_results_params.csv** holds the best configuration of forests for each dataset

**grid_results_combined_top_15** provides an easy to read overview of the best params for the top 15 scoring models


Turns out that parameters matter very little with the forests. There are dependencies like max_features and number_of_estimators (low -> low, high -> high), but overall there is no superior combination of params.


**ablation_results.txt** contains a short description of removed features and attained accuracy.

It can be seen that no single feature contributes an incredible amount. And if it's taken away, the model can make do with the other features still reasonably well. Most surprisingly, team ratings actually improve the accuracy if taken away. I'd attribute that to the fact that these ratings are done subjectively by experts as far I know, instead of being calculated by a perfect formula. 




The 'Normalized' and 'Reduced' descriptions in various files mean that the features used were normalized or their amount reduced using SelectKBest respectively.
