# Decision Tree

 We can think of a decision tree as a series of yes/no questions asked about our data eventually leading to a predicted class (or continuous value in the case of regression). This is an interpretable model because it makes classifications much like we do: we ask a sequence of queries about the available data we have until we arrive at a decision (in an ideal world).

 The technical details of a decision tree are in how the questions about the data are formed. In the CART algorithm, a decision tree is built by determining the questions (called splits of nodes) that, when answered, lead to the greatest reduction in Gini Impurity.

### Nodes

All the nodes, except the leaf nodes (colored terminal nodes), have 5 parts:
- Question asked about the data based on a value of a feature. Each question has either a True or False answer that splits the node. Based on the answer to the question, a data point moves down the tree.
- The Gini Impurity of the node. The average weighted Gini Impurity decreases as we move down the tree.
- Samples: The number of observations in the node.
- Value: The number of samples in each class. For example, the top node has 2 samples in class 0 and 4 samples in class 1.
- Class: The majority classification for points in the node. In the case of leaf nodes, this is the prediction for all samples in the node.

The leaf nodes do not have a question because these are where the final predictions are made.

### Gini Impurity

The Gini Impurity of a node is the probability that a randomly chosen sample in a node would be incorrectly labeled if it was labeled by the distribution of samples in the node.

- I<sub>g</sub>(n) = 1 - Σ<sub>i=1</sub><sup>J</sup>(p<sub>i</sub>)<sup>2</sup>

At each node, the decision tree searches through the features for the value to split on that results in the greatest reduction in Gini Impurity. It then repeats this splitting process in a greedy, recursive procedure until it reaches a maximum depth, or each node contains only samples from one class.

# Random Forrest

**Solution to Decision Tree Overfitting**

The random forest is a model made up of many decision trees. Rather than just simply averaging the prediction of trees (which we could call a “forest”), this model uses two key concepts that gives it the name random:
 1. Random sampling of training data points when building trees
 2. Random subsets of features considered when splitting nodes


 ### [Code Implementation](../Section%2020%20-%20Random%20Forest%20Classification/random_forest_code.py)
