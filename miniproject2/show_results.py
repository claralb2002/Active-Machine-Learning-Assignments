import pandas as pd
from modules.visualize import *

res = pd.read_csv('AL_results.csv', index_col='Unnamed: 0')
baseline = pd.read_csv('AL_results_baseline.csv', index_col='Unnamed: 0')

plot_committee_performance(res,baseline)
plot_committee_boxplot(res,baseline)