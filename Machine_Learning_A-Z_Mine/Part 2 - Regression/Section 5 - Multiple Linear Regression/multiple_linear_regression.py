# Multiple Linear Regression

# %% codecell
# Importing the libraries
import sys
import os
os.environ['VIRTUAL_ENV']
sys.path.append('Part 2 - Regression/Section 5 - Multiple Linear Regression')

# %% codecell
# preprocess data
import data_preprocessing_template as preprocessed_data
x_train, x_test, y_train, y_test, x, y  =  preprocessed_data.preprocess_data()

# %% codecell
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# %% codecell
# Predicting the Test set results
y_pred = regressor.predict(x_test)
