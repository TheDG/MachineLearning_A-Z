# Random Forest Regression

# %% codecell
# Importing the libraries
from sklearn.ensemble import RandomForestRegressor
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
if '__file__' not in globals():
    sys.path.append(os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 2 - Regression/Section 9 - Random Forest Regression')
import data_preprocessing_template as preprocessed_data

# %% codecell
# preprocess data
x, y = preprocessed_data.preprocess_data()

# %% codecell
# Fitting Random Forest Regression to dataset
regresor = RandomForestRegressor(criterion="mse", n_estimators=300, random_state=0)
regresor.fit(x, y.ravel())


# %% codecell
"""
# Visualising the Random Forest Regression results
plt.scatter(x, y, color='red')
plt.plot(x, regresor.predict(x), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

# %% codecell
# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, regresor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %% codecell
# Predicting a new result with Random Forest Regression
regresor.predict([[6.5]])

# xi = sc_x.fit_transform([[6.5]])
# sc_y.inverse_transform(regresor.predict(xi))
