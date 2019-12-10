# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv(
    '/Users/diegosinay/Downloads/ML/Machine Learning A-Z Mine/Part 1 - Data Preprocessing/Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#check missing data
df.isnull().sum()

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Replace missing data with mean value
#df['col name'] = df['col name'].fillna(df.mode()['col name'][0])

# Drop full rows with null values
#df = df.dropna(subset=['col name'])

# Drop full Col
#df = df.drop('col name', axis=1)
