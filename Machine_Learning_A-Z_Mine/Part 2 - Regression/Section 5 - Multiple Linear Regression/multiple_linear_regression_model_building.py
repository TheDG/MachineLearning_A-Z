# Multiple Linear Regression

# %% codecell
# Importing the libraries
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import sys
import numpy as np
sys.path.append(
    'Part 2 - Regression/Section 5 - Multiple Linear Regression')
import data_preprocessing_template as preprocessed_data

# %% codecell
# preprocess data
x_train, x_test, y_train, y_test, x, y = preprocessed_data.preprocess_data()


# %% codecell
# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# %% codecell
# Predicting the Test set results
y_pred = regressor.predict(x_test)

# %% codecell
# backwords elimination

# add constant to function
x = np.append(np.ones((50, 1), dtype=np.int), x, axis=1)

# %% codecell
# first iteration
X_optimal = x[:, [0, 1, 2, 3, 4, 5]].astype(float)
regressor_OLS = sm.OLS(y, X_optimal).fit()
regressor_OLS.summary()

# %% codecell
# 2nd iteration
X_optimal = x[:, [0, 1, 3, 4, 5]].astype(float)
regressor_OLS = sm.OLS(y, X_optimal).fit()
regressor_OLS.summary()

# %% codecell
# 3rd iteration
X_optimal = x[:, [0, 3, 4, 5]].astype(float)
regressor_OLS = sm.OLS(y, X_optimal).fit()
regressor_OLS.summary()

# %% codecell
# 4rd iteration / end
X_optimal = x[:, [0, 3, 5]].astype(float)
regressor_OLS = sm.OLS(y, X_optimal).fit()
regressor_OLS.summary()

# %% codecell
# automatic backward elmination

def backwardElimination(x, SL):
    numVars = len(x[0])
    selected_cols = list(range(0, numVars))
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars - 1):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    temp_col = selected_cols.pop(j)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        selected_cols.insert(j, temp_col)
                        print(regressor_OLS.summary())
                        return x_rollback, selected_cols
                    else:
                        continue
    regressor_OLS.summary()
    print(regressor_OLS.summary())
    selected_cols
    return x, selected_cols

SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]].astype(float)
X_Modeled, selected_cols = backwardElimination(X_opt, SL)
selected_cols
