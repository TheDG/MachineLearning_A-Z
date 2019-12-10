# %% md
# importing libaries

# %% codecell

import numpy as np
import pandas as pd
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# %% codecell
# importing dataset
df = pd.read_csv(
    '/Users/diegosinay/Downloads/ML/Machine Learning A-Z Mine/Part 1 - Data Preprocessing/Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
df

# %% codecell
# check missing data
df.isnull().sum()

# Take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# %% codecell
# Take care of categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encoding x data
# when categorical data has an implicit order
x1 = x.copy()
labelencoder_x = LabelEncoder()
x1[:, 0] = labelencoder_x.fit_transform(x1[:, 0])

# dummy encoding --> when their is no order to categories
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
x.astype(int)

# Encoding Y data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x = x.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_train
