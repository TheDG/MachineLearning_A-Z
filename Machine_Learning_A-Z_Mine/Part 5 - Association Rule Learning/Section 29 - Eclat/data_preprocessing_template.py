# %% md
# importing libaries

# %% codecell
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
        path = os.getcwd() + '/Market_Basket_Optimisation.csv'
    else:
        path = os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 5 - Association Rule Learning/Section 29 - Eclat/Market_Basket_Optimisation.csv'
    df = pd.read_csv(path, header = None)
    # df = df.replace({np.nan : 'nan12'})
    return df


# %% codecell
# Encode Categorial Data
def encode_categorical(df):
    categorical_feature_mask = df.dtypes==object
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    df[categorical_cols] = df[categorical_cols].astype('category')
    for col in categorical_cols:
        df[col] = df[col].cat.codes.astype('category')

    # df = df.replace({-1 : ''})
    return df

"""
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
# main
def preprocess_data():
    df = import_dataset()
    df = encode_categorical(df)
    # x_train, x_test, y_train, y_test = train_dataset(x, y)
    # x_train, x_test, sc_x = scale_data(x_train, x_test)
    # return x_train, x_test, y_train, y_test
    df.to_csv('Market_Basket_Optimisation.txt', sep=' ', index=False, header=False)


# %% codecell
if __name__ == '__main__':
    preprocess_data()
