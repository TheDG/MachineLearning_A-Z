# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv(
    '/Users/diegosinay/Downloads/ML/Machine Learning A-Z Mine/Part 1 - Data Preprocessing/Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encoding x data
# when categorical data has an implicit order
x1 = x.copy()
labelencoder_x = LabelEncoder()
x1[:, 0] = labelencoder_x.fit_transform(x1[:, 0])

# fabi oway of enconding cat data with order
#df['col'] = df['col'].apply(lambda value: int(value[0]) - 1)

# dummy encoding --> when their is no order to categories
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
x.astype(int)

# fabio way of dumy encondig
#def apply_dummy(df, cat, drop_first=True):
#    return pd.concat([df, pd.get_dummies(df[cat], prefix=cat, drop_first=drop_first)], axis=1).drop(cat, axis=1)
#for cat in dummy_features:
    #df = apply_dummy(df, cat)

# Encoding Y data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Fabio way of enconding binary data
#df['col'] = df['col'].apply(lambda value: 1 if value == 'Y' else 0)
