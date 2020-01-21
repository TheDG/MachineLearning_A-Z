# Data Prepocessing For Natural Text

# %% codecell
# importing libaries
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import re
import nltk
nltk.download('stopwords')


# %% codecell
# importing dataset
def import_dataset():
    if '__file__' in globals() or os.path.basename(os.getcwd()) != 'MachineLearning_A-Z':
        path = os.getcwd() + '/Restaurant_Reviews.tsv'
    else:
        path = os.getcwd() + '/Machine_Learning_A-Z_Mine/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv'
    df = pd.read_csv(path, delimiter='\t', quoting=3)
    return df


# %% codecell
# Cleaning the texts
def clean_text(df):
    corpus = []
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    for i in range(0, df.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word)
                  for word in review if not word in stopwords_set]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


# %% codecell
# Creating the Bag of Words model
def create_bag_words(corpus):
    cv = CountVectorizer(max_features=1500)
    return cv.fit_transform(corpus).toarray()


# %% codecell
# Splitting the dataset into the Training set and Test set
def train_dataset(x, y):
    return train_test_split(x, y, test_size=.15, random_state=0)


# %% codecell
def preprocess_data():
    df = import_dataset()
    corpus = clean_text(df)
    x = create_bag_words(corpus)
    y = df.iloc[:, 1].values
    return train_dataset(x, y)


# %% codecell
# main
if __name__ == '__main__':
    preprocess_data()
