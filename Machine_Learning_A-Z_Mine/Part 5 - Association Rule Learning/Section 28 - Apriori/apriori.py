# Apriori

# %% codecell
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# pd.options.display.html.table_schema = True
# pd.options.display.max_rows = None


# %% codecell
# Data Preprocessing
df = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
df.shape
for i in range(0, df.shape[0]):
    transactions.append([str(df.values[i,j]) for j in range(0, df.shape[1])])


# %% codecell
# Training Apriori on the dataset | Setting the paramters is experimental! Try to fit to your data
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.3, min_length = 2, min_lift = 3)


# %% codecell
# Visualising the results
results = list(rules)
df2 = pd.DataFrame(results).iloc[:,[2]]
# df2 = df2[['ordered_statistics']]
df3 = df2.applymap(lambda x: x[0])
df3 = df3.iloc[:, 0]
df3 = pd.DataFrame(list(df3), columns = ['items_base', 'items_add', 'confidence', "lift"])
df4 = pd.DataFrame(results).iloc[:,[1]]
final = pd.merge(df3, df4, right_index=True, left_index=True)
final
