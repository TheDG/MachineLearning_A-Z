# %% md

# %% codecell
# importing libaries
import pandas as pd
import os
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# pd.options.display.html.table_schema = True
# pd.options.display.max_rows = None


# %% codecell
# importing dataset
def import_dataset():
    if '__file__' in globals() or os.path.basename(os.getcwd()) != 'MachineLearning_A-Z':
        path = os.getcwd() + '/Ads_CTR_Optimisation.csv'
    else:
        path = os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv'
    df = pd.read_csv(path)
    df
    return df


"""
# Encode Categorial Data
def encode_categorical(x):
    ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    return x[:, 1:]

# Splitting the dataset into the Training set and Test set
def train_dataset(x, y):
    return train_test_split(x, y, test_size = .25, random_state = 0)

# Feature Scaling
def scale_data(x_train, x_test):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    #sc_y = StandardScaler()
    #y = y.reshape(-1, 1)
    #y = sc_y.fit_transform(y)
    return [x_train, x_test, sc_x]
"""


# %% codecell
def preprocess_data():
    x = import_dataset()
    # x = encode_categorical(x)
    # x_train, x_test, y_train, y_test = train_dataset(x, y)
    # x_train, x_test, sc_x = scale_data(x_train, x_test)
    # return x_train, x_test, y_train, y_test
    return x


# %% codecell
# main
if __name__ == '__main__':
    preprocess_data()
