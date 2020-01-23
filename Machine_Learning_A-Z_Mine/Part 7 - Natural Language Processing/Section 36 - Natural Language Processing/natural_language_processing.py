# Natural Language Processing

# %% codecell
# Importing the libraries
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tabulate import tabulate
import data_preprocessing as preprocessed_data
import pandas as pd
import sys
import os
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing')


# %% codecell
# Preprocess the data before algorithm is applied
x_train, x_test, y_train, y_test = preprocessed_data.preprocess_data()


# %% codecell
# Create Presentation Table Structure
columns = ['Accuracy', 'Precision', 'Sensitivity', 'F1 Score']
index = []
data = []


def add_to_table(cm, algorithm, data, index):
    index.append(algorithm)
    tn, fp, fn, tp = cm.ravel()
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
    precision = round(tp / (tp + fp), 3)
    sensitivity = round(tp / (tp + fn), 3)
    f1_score = round(2.0 * precision * sensitivity /
                     (precision + sensitivity), 3)
    data.append([accuracy, precision, sensitivity, f1_score])


# %% codecell
# Iterate over models
names = ["Naive Bayes", "Logistic Regression", "KNN", "SVM-linear",
         "SVM-RBF", "Random Forest", "Quadratic Discriminant Analysis"]
classifiers = [
    GaussianNB(),
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
    SVC(kernel="linear", random_state=0),
    SVC(kernel="rbf", random_state=0),
    RandomForestClassifier(
        n_estimators=20, criterion='entropy', random_state=0),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    add_to_table(cm, name, data, index)

# %% codecell
df = pd.DataFrame(data, columns=columns, index=names)
print(tabulate(df, headers="keys", tablefmt="psql"))
