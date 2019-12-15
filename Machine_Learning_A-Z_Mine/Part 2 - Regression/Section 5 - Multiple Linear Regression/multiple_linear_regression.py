# Multiple Linear Regression

# %% codecell
# Importing the libraries
from sklearn.linear_model import LinearRegression
import sys
import os
import numpy as np
if '__file__' not in globals():
    sys.path.append(os.getcwd() + 'Machine_Learning_A-Z_Mine/Part 2 - Regression/Section 5 - Multiple Linear Regression')
import data_preprocessing_template as preprocessed_data

# %% codecell
# preprocess data
x_train, x_test, y_train, y_test, x, y  =  preprocessed_data.preprocess_data()

# %% codecell
# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# %% codecell
# Predicting the Test set results
y_pred = regressor.predict(x_test)
