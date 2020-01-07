# Random Selection


# %% codecell
# Importing the libraries
import random
import matplotlib.pyplot as plt
import data_preprocessing_template as preprocessed_data
import sys
import os
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)')


# %% codecell
# Importing the dataset
df = preprocessed_data.preprocess_data()


# %% codecell
# Implementing Random Selection
df.shape
N = df.shape[0]
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = df.values[n, ad]
    total_reward = total_reward + reward


# %% codecell
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
