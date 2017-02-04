import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('experiment_logs/FD_paper-20161212-all_manual_analysis.csv')

plt.figure()
sns.set(font_scale=1.5)

r = df[df.algo.str.endswith('C')]
g = sns.FacetGrid(r, row='SFS', col='algo',
                  margin_titles=True, xlim=(-0.1, 1.1))
g.map(sns.boxplot, 'test_accuracy', 'dataset',
      whis=np.inf, color='c').set(xticks=[0.0, 0.5, 1.0])
g.map(sns.stripplot, 'test_accuracy', 'dataset', jitter=True,
      size=3, color='.3', linewidth=0).set(xticks=[0.0, 0.5, 1.0])
g.fig.subplots_adjust(hspace=.05, wspace=.05)
sns.despine(trim=True)
plt.savefig('figs/fig_classifier_test_accuracy.pdf',
            format='pdf', bbox_inches='tight')

r = df[df.algo.str.endswith('R')]
g = sns.FacetGrid(r, row='SFS', col='algo',
                  margin_titles=True, xlim=(-0.1, 1.1))
g.map(sns.boxplot, 'test_R2_score', 'dataset',
      whis=np.inf, color='c').set(xticks=[0.0, 0.5, 1.0])
g.map(sns.stripplot, 'test_R2_score', 'dataset', jitter=True,
      size=3, color='.3', linewidth=0).set(xticks=[0.0, 0.5, 1.0])
g.fig.subplots_adjust(hspace=.05, wspace=.05)
sns.despine(trim=True)
plt.savefig('figs/fig_regression_test_R2_score.pdf',
            format='pdf', bbox_inches='tight')

# CDF of accuracy for each of the classifier models, with and without SFS.
r = df[df.algo.str.endswith('C')]
g = sns.FacetGrid(r, row='SFS', col='algo',
                  margin_titles=True, xlim=(-0.1, 1.1))
g.map(sns.distplot, 'test_accuracy', hist_kws=dict(
    cumulative=True), kde_kws=dict(cumulative=True), rug=True)
g.fig.subplots_adjust(hspace=.15, wspace=.05)
sns.despine(trim=True)
plt.savefig('figs/fig_regression_CDF_test_accuracy.pdf',
            format='pdf', bbox_inches='tight')

# CDF of R^2 score for each of the regression models, with and without SFS.
r = df[df.algo.str.endswith('R')]
g = sns.FacetGrid(r, row='SFS', col='algo',
                  margin_titles=True, xlim=(-0.1, 1.1))
g.map(sns.distplot, 'test_R2_score', hist_kws=dict(cumulative=True),
      kde_kws=dict(cumulative=True), rug=True, bins=200)
g.fig.subplots_adjust(hspace=.15, wspace=.05)
sns.despine(trim=True)
plt.savefig('figs/fig_regression_CDF_test_R2_score.pdf',
            format='pdf', bbox_inches='tight')

# CDF of performance for each of the classification models, with and
# without SFS.
r = df[df.algo.str.endswith('C')]
g = sns.FacetGrid(r, row='SFS', col='algo', margin_titles=True)
g.map(sns.distplot, 'runtime_seconds', hist_kws=dict(cumulative=True),
      kde_kws=dict(cumulative=True), rug=True, bins=200)
g.fig.subplots_adjust(hspace=.05, wspace=.05)
g.add_legend()
plt.show()

# CDF of performance for each of the classification models, with and
# without SFS.
r = df[df.algo.str.endswith('C')]
g = sns.FacetGrid(r, row='SFS', col='algo', margin_titles=True)
g.map(sns.distplot, '', hist_kws=dict(cumulative=True),
      kde_kws=dict(cumulative=True), rug=True, bins=200)
g.fig.subplots_adjust(hspace=.05, wspace=.05)
g.add_legend()
plt.show()

# Only rows where test_accuracy is higher than 0.5:
# TODO: limit yticks form 0 to 1.
r = df[(df.algo.str.endswith('C')) & (df.test_accuracy > 0.5)]
g = sns.FacetGrid(r, row='target_is_numerical', col='algo', margin_titles=True)
g.map(sns.regplot, 'target_num_unique', 'test_accuracy', order=2)
g.fig.subplots_adjust(hspace=.05, wspace=.05)
g.add_legend()
plt.show()

# Only rows where test_R2_score is higher than 0.5:
# TODO: limit yticks from 0 to 1.
r = df[(df.algo.str.endswith('R')) & (df.test_R2_score > 0.5)]
g = sns.FacetGrid(r, row='target_is_continuous',
                  col='algo', margin_titles=True)
g.map(sns.regplot, 'target_num_unique', 'test_R2_score', order=2)
g.fig.subplots_adjust(hspace=.05, wspace=.05)
g.add_legend()
plt.show()

df[(df['algo'] == 'LinR') & (df['SFS'] == False)].groupby(
    ['dataset']).mean()[['num_rows', 'runtime_seconds']]
