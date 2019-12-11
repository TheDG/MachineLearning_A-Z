# %% md
# importing libaries

# %% codecell
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# %% codecell
# importing dataset
def import_dataset():
    df = pd.read_csv(
        '/Users/diegosinay/GitHub/MachineLearning_A-Z/Machine_Learning_A-Z_Mine/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y


# %% codecell
# Encode Categorial Data
def encode_categorical(x):
    ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    return x[:, 1:]


# %% codecell
# Splitting the dataset into the Training set and Test set
def train_dataset(x, y):
    return train_test_split(x, y, test_size = .2, random_state = 0)

# %% codecell
# main
def preprocess_data():
    x, y = import_dataset()
    x = encode_categorical(x)
    x_train, x_test, y_train, y_test = train_dataset(x, y)
    # no need for data scalling
    return [x_train, x_test, y_train, y_test, x, y]

# %% codecell
if __name__ == '__main__':
    preprocess_data()
