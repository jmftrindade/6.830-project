import pandas as pd
import numpy as np
import random

data = pd.read_csv("~/Desktop/red.csv", delimiter=';')
print data
print data.head()

ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
print ix
 
print data.dtypes

data['quality']=data['quality'].astype(float)
print data.dtypes

for row, col in random.sample(ix, int(round(.5*len(ix)))):
  data.iat[row, col] = np.nan

print data
 
print data.head()

data.to_csv('~/Desktop/testM.csv', sep=';')

data1=pd.read_csv('~/Desktop/testM.csv',delimiter=';')
print data.head()

print data1.head()

result = pd.concat([data, data], axis=1)
print result.head()

print data.columns

new_column_names = []
for column in data:
    new_column_names.append(column)

new_column_names
['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
new_column_names = []
for column in data:
     new_column_names.append(column + '_2')
 
new_column_names
['fixed acidity_2', 'volatile acidity_2', 'citric acid_2', 'residual sugar_2', 'chlorides_2', 'free sulfur dioxide_2', 'total sulfur dioxide_2', 'density_2', 'pH_2', 'sulphates_2', 'alcohol_2', 'quality_2']
data2 = data
data2.columns = new_column_names
data2.head()
  
concatenated = pd.concat([data, data2], axis=1)
concatenated.head()

data.head()
  
data1 = pd.read_csv('~/Desktop/testM.csv', sep=';')
data2 = pd.read_csv('~/Desktop/red.csv', sep=';')
data2 = pd.read_csv('~/Desktop/red.csv', sep=';')
concatenated = pd.concat([data1, data2], axis=1)
concatenated.head()

new_column_names
['fixed acidity_2', 'volatile acidity_2', 'citric acid_2', 'residual sugar_2', 'chlorides_2', 'free sulfur dioxide_2', 'total sulfur dioxide_2', 'density_2', 'pH_2', 'sulphates_2', 'alcohol_2', 'quality_2']
data2.columns = new_column_names
data2.head()
  
concatenated = pd.concat([data1, data2], axis=1)
concatenated.head()
 
concatenated.to_csv('~/Desktop/training.csv', sep=';')
concatenated.to_csv('~/Desktop/training_with_comma.csv', sep=',')
concatenated['target'] = concatenated[new_column_names].apply(tuple,axis=1)
concatenated.head()

concatenated.to_csv('~/Desktop/trainig_target.csv', sep=',')

