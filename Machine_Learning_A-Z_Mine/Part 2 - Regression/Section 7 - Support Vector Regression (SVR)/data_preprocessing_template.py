# %% md
# importing libaries

# %% codecell
import pandas as pd
import os
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None


# %% codecell
# importing dataset
def import_dataset():
    if '__file__' in globals() and os.path.basename(os.getcwd()) != 'MachineLearning_A-Z':
        path = os.getcwd() + '/Position_Salaries.csv'
    else:
        path = os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv'
    df = pd.read_csv(path)
    x = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values
    return x, y


# %% codecell
"""
# Encode Categorial Data
def encode_categorical(x):
    ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    return x[:, 1:]


# Splitting the dataset into the Training set and Test set
def train_dataset(x, y):
    return train_test_split(x, y, test_size = .2, random_state = 0)
"""
# %% codecell
# Feature Scaling
def scale_data(x, y):
    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)
    sc_y = StandardScaler()
    y = y.reshape(-1, 1)
    y = sc_y.fit_transform(y)
    return [x, y, sc_x, sc_y]


# %% codecell
# main
def preprocess_data():
    x, y = import_dataset()
    # x = encode_categorical(x)
    # x_train, x_test, y_train, y_test = train_dataset(x, y)
    x, y, sc_x, sc_y = scale_data(x, y)
    return x, y, sc_x, sc_y


# %% codecell
if __name__ == '__main__':
    preprocess_data()
