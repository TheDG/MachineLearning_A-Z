# Upper Confidence Bound

# %% codecell
# Importing the libraries
import math
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
# Implementing UCB
N = df.shape[0]
d = df.shape[1]
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) /
                                numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            """Select ads / arms in secuential manner until we have atleast
            one data point for each"""
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = df.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
total_reward


# %% codecell
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
