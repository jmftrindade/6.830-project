import pandas as pd
import numpy as np
import random

data1 = pd.read_csv("~/Desktop/red.csv", delimiter=';')

#data2 has one column, I chose density, you can checnge it to nay other column in red.csv
data2= data1['density']
data2= data2.rename("target")

#concat data with that one column 'data2'
data = pd.concat([data1, data2], axis=1)

ix = [row for row in range(data['density'].shape[0])]
#insert NaN randomly in target column
for row in random.sample(ix, int(round(.5*len(ix)))):
  data['density'].iat[row] = np.nan

print data.head()

data.to_csv('red_one_target2.csv', sep=',')


