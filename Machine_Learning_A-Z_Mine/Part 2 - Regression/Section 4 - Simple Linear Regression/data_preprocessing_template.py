# %% md
# importing libaries

# %% codecell
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# %% codecell
# importing dataset
def import_dataset():
    df = pd.read_csv(
        'Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
    x = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values
    # df
    # check missing data
    # df.isnull().sum()
    return x, y

# %% codecelL
# Splitting the dataset into the Training set and Test set
def train_dataset(x, y):
    return train_test_split(x, y, test_size = 1/3, random_state = 0)


# %% codecell
# main
def preprocess_data():
    x, y = import_dataset()
    return train_dataset(x, y)

# %% codecell
if __name__ == '__main__':
    preprocess_data()
