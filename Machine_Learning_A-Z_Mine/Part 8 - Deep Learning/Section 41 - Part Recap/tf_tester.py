# %% markdown
# Download and install the TensorFlow 2 package. Import TensorFlow into your program:
# %% codecell
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('rm -rf ./logs/')
# Helper libraries
import numpy as np
import datetime
import matplotlib.pyplot as plt

print(tf.__version__)

# %% markdown
# Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow:
# %% codecell
fashion_mnist = keras.datasets.fashion_mnist

# %% markdown
# Loading the dataset returns four NumPy arrays:
# %% codecell
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# %% markdown
# Each image is mapped to a single label. Since the *class names* are not included with the dataset, store them here to use later when plotting the images:
# %% codecell
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %% markdown
# ## Explore the data
# Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
# %% codecell
x_train.shape

# %% markdown
# ## Preprocess the data
# The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:
# %% codecell
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# %% markdown
# Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the *training set* and the *testing set* be preprocessed in the same way:
# %% codecell
x_train = x_train / 255.0
x_test = x_test / 255.0

# %% markdown
# To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the *training set* and display the class name below each image.
# %% codecell
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

# %% markdown
# ## Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.

# %% markdown
# ### Set up the layers
# The basic building block of a neural network is the *layer*. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
# Most of deep learning consists of chaining together simple layers. Most layers, such as `tf.keras.layers.Dense`, have parameters that are learned during training.
# %% codecell
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# %% markdown
# The first layer in this network, `tf.keras.layers.Flatten`, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
#
# After the pixels are flattened, the network consists of a sequence of two `tf.keras.layers.Dense` layers. These are densely connected, or fully connected, neural layers. The first `Dense` layer has 128 nodes (or neurons). The second (and last) layer is a 10-node *softmax* layer that returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
#
# ### Compile the model
#
# Before the model is ready for training, it needs a few more settings. These are added during the model's *compile* step:
#
# * *Loss function* —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# * *Optimizer* —This is how the model is updated based on the data it sees and its loss function.
# * *Metrics* —Used to monitor the training and testing steps. The following example uses *accuracy*, the fraction of the images that are correctly classified.
# %% codecell
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% markdown
# ## Train the model
#
# Training the neural network model requires the following steps:
#
# 1. Feed the training data to the model. In this example, the training data is in the `train_images` and `train_labels` arrays.
# 2. The model learns to associate images and labels.
# 3. You ask the model to make predictions about a test set—in this example, the `test_images` array.
# 4. Verify that the predictions match the labels from the `test_labels` array.
#
#
# %% markdown
# ### Feed the model
#
# To start training,  call the `model.fit` method—so called because it "fits" the model to the training data:

# %% markdown
# ### Using TensorBoard with Keras Model.fit()
# %% codecell
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, epochs=10,validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

# get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')

# %% markdown
# ### Evaluate accuracy
#
# Next, compare how the model performs on the test dataset:
# %% codecell
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# %% markdown
# ### Make predictions
# With the model trained, you can use it to make predictions about some images.
# %% codecell
predictions = model.predict(x_test)

# %% markdown
# Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
# %% codecell
predictions[0]

# %% markdown
# A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:
# %% codecell
np.argmax(predictions[0])

# %% markdown
# So, the model is most confident that this image is an ankle boot, or `class_names[9]`. Examining the test label shows that this classification is correct:
# %% codecell
y_test[0]

# %% markdown
# Graph this to look at the full set of 10 class predictions.
# %% codecell
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# %% markdown
# ### Verify predictions
#
# With the model trained, you can use it to make predictions about some images.
# %% markdown
# Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are blue and incorrect prediction labels are red. The number gives the percentage (out of 100) for the predicted label.
# %% codecell
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], y_test, x_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i],  y_test)
plt.show()
# %% codecell
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], y_test, x_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i],  y_test)
plt.show()

# %% markdown
# Let's plot several images with their predictions. Note that the model can be wrong even when very confident.
# %% codecell
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, x_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()

# %% markdown
# ## Use the trained model
# Finally, use the trained model to make a prediction about a single image.
# %% codecell
# Grab an image from the test dataset.
img = x_train[1]
print(img.shape)

# %% markdown
# `tf.keras` models are optimized to make predictions on a *batch*, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list:
# %% codecell
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)

# %% markdown
# Now predict the correct label for this image:
# %% codecell
predictions_single = model.predict(img)
print(predictions_single)

# %% codecell
plot_value_array(1, predictions_single[0], y_test)
_ = plt.xticks(range(10), class_names, rotation=45)

# %% markdown
# `model.predict` returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:
# %% codecell
np.argmax(predictions_single[0])
# %% markdown
# And the model predicts a label as expected.
