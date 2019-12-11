# Simple Linear Regression

# %% codecell
# Importing the libraries
import sys
import matplotlib.pyplot as plt
sys.path.append('Part 2 - Regression/Section 4 - Simple Linear Regression')

# %% codecell
# preprocess data
import data_preprocessing_template as preprocessed_data
x_train, x_test, y_train, y_test  =  preprocessed_data.preprocess_data()

# %% codecell
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# %% codecell
# Predicting the Test set results
y_pred = regressor.predict(x_test)

# %% codecell
# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %% codecell
# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
