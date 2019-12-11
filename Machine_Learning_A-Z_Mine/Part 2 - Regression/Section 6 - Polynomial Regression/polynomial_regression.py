# Polynomial Regression
# when simple linear regresion has parabolic shape

# %% codecell
# Importing the libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
if '__file__' not in globals():
    sys.path.append(os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 2 - Regression/Section 6 - Polynomial Regression')
import data_preprocessing_template as preprocessed_data

# %% codecell
# preprocess data
x, y = preprocessed_data.preprocess_data()

# %% codecell
# Fitting Simple Linear Regression to dataset
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# %% codecell
# Fitting Polynomial Linear Regression to dataset
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


# %% codecell
# Visualising the Linear Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %% codecell
# Visualising the Polynomial Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg2.predict(x_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %% codecell
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %% codecell
# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
