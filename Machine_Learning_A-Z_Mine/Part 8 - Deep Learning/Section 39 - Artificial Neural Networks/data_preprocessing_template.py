# %% md
# Data Prepocessing Template

# %% codecell
# importing libaries
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# pd.options.display.html.table_schema = True
# pd.options.display.max_rows = None


# %% codecell
# importing dataset
def import_dataset():
    if '__file__' in globals() or os.path.basename(os.getcwd()) != 'MachineLearning_A-Z':
        path = os.getcwd() + '/Churn_Modelling.csv'
    else:
        path = os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks/Churn_Modelling.csv'
    df = pd.read_csv(path)
    x = df.iloc[:, 3:13].values
    y = df.iloc[:, 13].values
    df
    return x, y


# %% codecell
# Encode Categorial Data
def encode_categorical(x):
    # transform countries intro ints
    labelencoder_x1 = LabelEncoder()
    x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])
    # transform gender intro ints
    labelencoder_x2 = LabelEncoder()
    x[:, 2] = labelencoder_x2.fit_transform(x[:, 2])
    # create dummy variable for countries
    ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    # drop 1 column of dummy countries to not fall into dummy variable trap
    x = x[:, 1:]
    return x


# %% codecell
# Splitting the dataset into the Training set and Test set
def train_dataset(x, y):
    return train_test_split(x, y, test_size = .2, random_state = 0)


# %% codecell
# Feature Scaling
def scale_data(x_train, x_test):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    #sc_y = StandardScaler()
    #y = y.reshape(-1, 1)
    #y = sc_y.fit_transform(y)
    return [x_train, x_test, sc_x]


# %% codecell
def preprocess_data():
    x, y = import_dataset()
    x = encode_categorical(x)
    x_train, x_test, y_train, y_test = train_dataset(x, y)
    x_train, x_test, sc_x = scale_data(x_train, x_test)
    # return x_train, x_test, y_train, y_test
    return x_train, x_test, y_train, y_test


# %% codecell
# main
if __name__ == '__main__':
    preprocess_data()
