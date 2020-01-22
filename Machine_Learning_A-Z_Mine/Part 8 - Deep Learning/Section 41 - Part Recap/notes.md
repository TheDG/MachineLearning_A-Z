# Intro

Deep learning (also known as deep structured learning or hierarchical learning or differential programming) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Inspired by the structure and function of the brain called artificial neural networks. Also known as deep neural learning or deep neural network

-   Learning can be supervised, semi-supervised or unsupervised.
-   Are data and processing power intensive.

Neural nets are particularly useful when the relationship between the inputs and outputs may not be readily apparent to a human analyst
### Potential Applications

-   Artificial Neural Networks for Regression and Classification
-   Convolutional Neural Networks for Computer Vision
-   Recurrent Neural Networks for Time Series Analysis
-   Self Organizing Maps for Feature Extraction
-   Deep Boltzmann Machines for Recommendation Systems
-   Auto Encoders for Recommendation Systems

# Intuition

Try to recreate neural networks as in the brain.

-   Why: One of the most powerful learning tools

Basic concept is that we have an input layer (of neurons), that is connected with a vast hidden layer (of neurons), that interact in ways that we dont understand, but affect the output layer(neurons). This output value is what we are trying to predict.

# The Neuron

### Structure

-   Neuron: Main body
-   Dendrites: Branches --> Receivers
-   Axon: Long tail --> Emitters
-   Connection between Dendrites and Axon is called synapse

### Machine Representation

 A Neuron has input layers (values) and a output signal (value)

**Input layers**
- Neurons can have input layers from other hidden layers / neurons
- In a brain, the input values are your senses
- Input values are your independent variables
- Input variables have to be standardized(mean 0, variance 1) or normalize.

**Output layers**
- Can be continuous, binary or categorical

**Synapses**
- Get assigned weights --> CRUCIAL since its the way the NN learns. Element that gets adjusted to represents learning.

**Neuron**
1. All values get added up (Weighted sum)
2. Applies activation function to weighted sum and decides if signal is passed on or not

# The Activation Function
We are going to cover the 4 Predominant Types

1. **Threshold Function:** Binary function
  - Depending if weighted sums surpass a certain limit, value becomes true.
2. **Sigmoid Function:** Smooth and gradual function
  - f(x) = 1 / (1+e<sup>-x</sup>)
3. **Rectifier Function:** One of the most used
  - f(n) = max(x, 0)
4. **Hyperbolic Tangent:** Similar to Sigmoid, but values go from -1 to 1
  - f(n) = (1 - e<sup>-2x</sup>) / (1+e<sup>-2x</sup>)

The activation functions do not have to be the same for the hidden layers and the output layer. A typical use case is applying the rectifier function for the hidden layers and the sigmoid function for the output layer

### For Binary Dependent variables
Best functions would be the Threshold and the Sigmoid function.

# How do NN Work
NN work by combining inputs and creating new inputs / neurons. These neuron then represent new specific predictors, that can be very powerful if correctly trained.

# How do NN Learn
In NN you provide inputs and specify outputs, and you let the program figure it out (learn) on its own.

**Perceptron** = Single Layer NN

**Epoch:** Train a NN over a full dataset

**Cost Function:** Function used to evaluate a perceptrons accuracy. Compares actual and estimated values
  - Typical Function = .5(actual - estimated)<sup>2</sup>

Data is feed into the system and the cost functions are evaluated (Cost functions are summated for each neuron). Then feedback is sent back into the system, and the weights are adjusted to represent that. This process is repeated for the same set of data various times, trying to minimize the cost function. --> **Backpropagation**

# Gradient Descent
**Curse of Dimensionality:** Solving Discrete Problems / Optimizations are not efficient!
  - Reason why brute force to find weights for optimizing does not work

Gradient Descent is the way how weights are adjusted (minimize cost function)
- **Differentiate and follow the path with the biggest (negative) slope**. In this case we feed all the data to the NN, and we adjust weights until finding solution.
- Only works for convex problems

### Stochastic Gradient Descent
Gradient descent finds local optimal values, but can fail if problem is not convex (gets stuck in local optimum).

Instead of running the data as a batch through the NN, we run the data per row (or smaller groups), and adjust the weights for each row.

### Stochastic vs Non-Stochastic Gradient Descent
- Stochastic has higher fluctuations (since it does a row at a time), hence is more likely to find global over local optimum.
- Stochastic is actually faster since it has to load less data into the system to perform operations!
- Non-Stochastic has the advantage of being a deterministic method.

# Backpropagation
Errors (difference between estimated and actual values | cost function) are  propagated back through the network in order to learn by adjusting the weights.

- Complex algorithm that allows all weights to be adjusted simultaneously

Backpropagation is a method to adjust the connection weights to compensate for each error found during learning. The error amount is effectively divided among the connections. Technically, backprop calculates the gradient (the derivative) of the cost function associated with a given state with respect to the weights. The weight updates can be done via stochastic gradient descent or other methods

# Training the ANN with Stochastic Gradient Descent Alogirthm
    1. Randomly initialize the weights to small numbers close to 0
    2. Input the first observation of you dataset in the input layer, each feature in one input node
    3. Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting to the output layer (y)
    4. Compare the predicted result to the actual result (calculate error)
    5. Back-Propagation: from right to left, the error is back-propagated.
      * Update the weights according to how much they are responsible for the error. The learning rate decides how much we update the weights
    6. Repeat steps 1-5 and update weights
      * If update weights after each observation --> Reinforcement Learning
      * If update weights after a batch of observation --> Batch Learning
    7. When the whole training set passed through the ANN, that makes an epoch.

# [TF 2.0 Basic classification: Classify images of clothing](tf_tester.ipynb)
