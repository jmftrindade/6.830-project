## An auto-cleaning layer for data management systems

Class project for Fall 2016's edition of [MIT's 6.830/6.814: Database Systems](http://db.csail.mit.edu/6.830/).

In this project, we proposed and evaluated the performance -- both prediction accuracy as well as time to train -- of different supervised learning algorithms on the task of predicting missing values in tabular data.  Additionally, we purposefully relied on an auto-ML like approach, where we didn't perform any explicit feature engineering, and instead used SFS (Sequential Feature Selection) to prune the search space of features to consider.

The tl;dr was: random forests provided the best trade-off in terms of prediction accuracy and time to train.  We also found that using SFS added incurred in significant performance penalty at training time, while offering only marginal gains on prediction accuracy compared to using all available table columns as features.

Final report available [here](report.pdf).

### TODO
- [ ] Add at least some documentation on how the scripts are organized, and how to run them.
