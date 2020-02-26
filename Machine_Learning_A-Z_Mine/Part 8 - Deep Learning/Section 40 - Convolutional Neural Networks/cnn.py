# %% markdown
# Convolutional Neural Network

# %% codecell
# Importing the libraries
# from tabulate import tabulate
# from sklearn.metrics import confusion_matrix
# import pandas as pd
import datetime
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import image_preprocessing_template as preprocessed_images

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('rm -rf ./logs/')
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)vertical_flip')
print(tf.__version__)

# Part 1 - Building the CNN
# %% codecell
# Initialising the CNN
classifier = Sequential()

# %% codecell
# Step 1 - Convolution
# Conv2D(# of feature detectors, Matrix Size(r, c), input Shape(r, c, layers))
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3),
                      data_format="channels_last", activation='relu'))

# %% codecell
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# %% codecell
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# %% codecell
# Step 3 - Flattening
classifier.add(Flatten())

# %% codecell
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# %% codecell
# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tf.keras.utils.plot_model(
    classifier,
    to_file='model.png',
    show_shapes=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96
)

# %% codecell
# Part 2 - Preprocess and Augment the Images
training_set, test_set = preprocessed_images.preprocess_images()


# Part 3 - Fitting the CNN to the images
# To see tensorflow board, run tensorboard --logdir logs/fit in a dif. terminal
# %% codecell
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)
classifier.fit(training_set,
               steps_per_epoch=8000 // 32,
               epochs=25,
               validation_data=test_set,
               validation_steps=2000 // 32,
               callbacks=[tensorboard_callback])
