# SVR

# %% codecell
# Importing the libraries
from sklearn.svm import SVR
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
if '__file__' not in globals():
    sys.path.append(os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)')
import data_preprocessing_template as preprocessed_data

# %% codecell
# preprocess data
x, y, sc_x, sc_y = preprocessed_data.preprocess_data()

# %% codecell
# Fitting SVR to dataset
regresor = SVR(kernel = 'rbf')
regresor.fit(x, y.ravel())


# %% codecell
# Visualising the SVR results
plt.scatter(x, y, color='red')
plt.plot(x, regresor.predict(x), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %% codecell
# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, regresor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %% codecell
# Predicting a new result with SVR
xi = sc_x.fit_transform([[6.5]])
sc_y.inverse_transform(regresor.predict(xi))
