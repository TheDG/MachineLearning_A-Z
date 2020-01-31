# Intro
A Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other

Basic Concept: Input Image --> CNN --> Output Label

Every image as a digital representation
- In a B/W image its a 2D array where each value is an int between 0 and 255
- In a Color image its a 3D array. Its 3 layers (Red, Green, Blue) of 2D arrays. These 3 layers give the name to RGB color representation. Each value is the intensity of that color for a b it

### Potential Applications
- Image and Video recognition
- Recommender systems
- Image classification
- Medical image analysis
- Natural language processing

CNN has various steps to it.

# 1) Convolution Operation
In mathematics (in particular, functional analysis) convolution is a mathematical operation on two functions (f and g) that produces a third function expressing how the shape of one is modified by the other. The term convolution refers to both the result function and to the process of computing it. It is defined as the integral of the product of the two functions after one is reversed and shifted.

**Primary Purpose:** Find features in image (using feature detector), putting them in a feature map, preserving spacial relation between pixels

### Elements
- Feature Detector = Kernel = Filter
  - Purpose: Detect certain features (integral parts)
  - Depends on the pattern on the detector
  - This feature detector mimics how we see. We look / detect patterns | elements | features
  - Get ride of non important information
- Feature Map = Convolved Feature = Activation Map
  - Your output to when you apply a convolution operator to something
  - Only contains important/relevant features

### Process (Image Processing)
- f is the image input and g is the feature detector.
- we apply matrix multiplication between f and g
- How many cols, rows the filter is shifted between operations is called the stride.
  - The bigger the stride, the more you reduce the size of the feature map --> reduce size
  - Some information is lost
- We create multiple feature maps --> we use various filters --> **Convolution Layer**
  - Helps preserve information
  - Network through training decides which features are important to identify each category

# 1.b) ReLU Layer (Rectified Linear Unit)
Applying rectifier function on-top of the convolution layer.
 - **Increase non-linearity**
  - Increases abrupt changes, instead of smooth transitions
 - Why: Images are non linear --> pixels, borders, colors, etc

# 2) Pooling (aka Downsampling)
A pooling layer is another building block of a CNN. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network. As it removes parameters and information, pooling helps prevent **overfitting**. Pooling layer operates on each feature map independently. The most common approach used in pooling is max pooling.

**Spacial Variance:** CNN does not care where feature is located or how --> able to identify distorted features --> have some flexibility.

### Process (Image Processing)
- Applied on the Feature Map
- In Max pooling we traverse the feature map (similar as in matrix multiplication) with a mask where the filter / mask we use is matrix that selects the highest value for that range
  - We need to select a matrix size and stride.

# 3) Flattening
Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector. And it is connected to the final classification model (usually ANN), which is called a fully-connected layer

# 4) Full Connection
Adding ANN to out Convolutional Network.
- Hidden layers are fully connected layers

Flattened layer becomes the input layer for the ANN

**Idea:** Combine features into more significant attributes that predict our labels / classes better.

**Output:**
- When its a regression we have a 1 node outputs
- In classification problems we have 1 Output Node per class / label.
- We can consider that the last layer in the fully connected layer (not the output), signify how probable the feature (or data aggregation) they represent belongs to a certain label.

**Backpropagation:** Weights are adjusted in the ANN, but also the feature detectors / filters are also adjusted --> how network is optimized / trained
- We could have been looking for the wrong features.

# Softmax & Cross-Entropy

**Question:** If output neurons are independent of each other, how do they the calibrate between each other so that probabilities adds up to 1? --> Softmax Function

**Softmax function**, aka, normalized exponential function, is a generalization of the logistic function, that "squashes" a k-dimensional vector of arbitrary real values into a k-dimensional vector of real values (in the range from 0 to 1), that adds up to one
- σ(z)<sub>i</sub> = e<sup>z<sub>i</sub></sup> / Σ<sub>j</sub><sup>K</sup>e<sup>z<sub>j</sub></sup> |  for i in(1,K) and z in (z<sub>1</sub>, z<sub>k</sub>)

**Cross Entropy:** Way of assessing the performance of the network (cost function). Works better than the Mean Squared Error (MSE) Function for CNN (after applying the SoftMax function) --> when you apply the Cross Entropy function to a CNN, its cold the loss function.
- Still want to minimize in order to maximize performance of network.
- H(p,q) = - Σ<sub>x</sub> p(x) × log<sub>q</sub>(x)
  - p(x): true distribution
  - q(x): estimated probability distribution
- Pros:
  - Log Factor helps adjust absolute impact of adjustments to relative impact --> learns faster
