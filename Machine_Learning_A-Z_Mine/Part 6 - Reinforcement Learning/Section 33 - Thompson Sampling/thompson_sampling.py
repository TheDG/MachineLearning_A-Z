# Thompson Sampling

# %% codecell
# Importing the libraries
import random
import matplotlib.pyplot as plt
import data_preprocessing_template as preprocessed_data
import sys
import os
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling')


# %% codecell
# Importing the dataset
df = preprocessed_data.preprocess_data()


# %% codecell
# Implementing Thompson Sampling
N = df.shape[0]
d = df.shape[1]
ads_selected = []
numbers_of_rewards_S = [0] * d
numbers_of_rewards_F = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_S[i] + 1, numbers_of_rewards_F[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        numbers_of_rewards_S[ad] += 1
    else:
        numbers_of_rewards_F[ad] += 1
    total_reward = total_reward + reward
total_reward


# %% codecell
# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
