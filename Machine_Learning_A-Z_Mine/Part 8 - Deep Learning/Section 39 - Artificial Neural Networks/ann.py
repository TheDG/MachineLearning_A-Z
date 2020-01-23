# %% markdown
# Artificial Neural Network

# %% codecell
# Importing the libraries
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import datetime
import data_preprocessing_template as preprocessed_data
import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('rm -rf ./logs/')
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks')
print(tf.__version__)

# %% codecell
# Importing the dataset
x_train, x_test, y_train, y_test = preprocessed_data.preprocess_data()

# %% markdown
# Part 2 - Now let's make the ANN!

# %% codecell
# Initialising the ANN
classifier = Sequential()

# %% markdown
# When you are not sure / do not want to be an artist, choose units / output as
# the average between the input node and output nodes
# %% codecells
# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform',
                     activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# %% codecell
# Compiling the ANN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% codecell
# Fitting the ANN to the Training set
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)
classifier.fit(x_train, y_train, batch_size=5, epochs=100,
               validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

# %% markdown
# Part 3 - Making the predictions and evaluating the model
# %% codecell
# Predicting the Test set results
y_predTest = classifier.predict(x_test)
y_predTest = (y_predTest > 0.5)

# Making the Confusion Matrix for test set
cm = confusion_matrix(y_test, y_predTest)
df = pd.DataFrame(cm, columns=['Negative Pred', 'Positive Pred'], index=[
                  'Neg Data', 'Pos Data'])
print("Test Data confusion_matrix")
print(tabulate(df, headers="keys", tablefmt="psql"))

# Making the Confusion Matrix for train set
y_predTrain = classifier.predict(x_train)
y_predTrain = (y_predTrain > 0.5)
cm2 = confusion_matrix(y_train, y_predTrain)


# %% codecell
def add_to_table(cm, algorithm, data, names):
    names.append(algorithm)
    tn, fp, fn, tp = cm.ravel()
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
    precision = round(tp / (tp + fp), 3)
    sensitivity = round(tp / (tp + fn), 3)
    f1_score = round(2.0 * precision * sensitivity /
                     (precision + sensitivity), 3)
    data.append([accuracy, precision, sensitivity, f1_score])


# %% codecell
# Accuracy Comparison | Check Fit
columns = ['Accuracy', 'Precision', 'Sensitivity', 'F1 Score']
names = []
data = []
add_to_table(cm, 'ANN (test)', data, names)
add_to_table(cm2, 'ANN (train)', data, names)
df2 = pd.DataFrame(data, columns=columns, index=names)
print(tabulate(df2, headers="keys", tablefmt="psql"))
