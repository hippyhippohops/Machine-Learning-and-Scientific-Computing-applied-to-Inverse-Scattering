import matplotlib.pyplot
import numpy as np
import pandas as pd
import seaborn as sns;

sns.set()
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import keras
import matplotlib.pyplot as plt

# Here, we create a simple class to implement our perceptron.
"""
We know that our input data has 2 pieces of input (the coordinates in our graph) and a binary output (the type of data
point), distinguished by different colors. 
"""

def sigmoid(s):
    # Activation function
    return 1 / (1 + np.exp(-s))


def sigmoid_prime(s):
    # Derivative of the Sigmoid
    return sigmoid(s) * (1 - sigmoid(s))


class FFNN(object):

    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        # Adding 1 as it will be our bias
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size

        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0

        # The whole weight matrix, from the inputs till the hidden layer
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        # The Final set of weights from the hidden layer till the output layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        # Forward propgation through our network
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.z1 = np.dot(X, self.w1)  # dot product of X (inout) and first set of 3x2 weights
        self.z2 = sigmoid(self.z1)  # Activation Function
        self.z3 = np.dot(self.z2, self.w2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = sigmoid(self.z3)  # final activation function
        return o

    def predict(self, X):
        return forward(self, X)

    def backward(self, X, y, output, step):
        # Backward propagation of the errors
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.o_error = y - output  # error in output
        self.o_delta = self.o_error * sigmoid_prime(output) * step  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(
            self.w2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2) * step  # Applying derivative of sigmoid to z2 error

        self.w1 += X.T.dot(self.z2_delta)  # Adjusting first of weights
        self.w2 += self.z2.T.dot(self.o_delta)  # Adjusting the second set of weights

    def fit(self, X, y, epochs=10, step=0.05):
        for epoch in range(epochs):
            X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
            output = self.forward(X)
            self.backward(X, y, output, step)


def main():
    """
    Here, we create a dataset. We will do so by sampling from 2 distinct normal distributions that we created, labeling
    the data according to the distribution.
    """
    # Inintialisg random number
    np.random.seed(11)  # FIND OUT WHAT RANDOM FUNCTION DOES

    ### Creating the dataset

    # mean and standard deviation for the x belonging to the first class
    mu_x1, sigma_x1 = 0, 0.1

    # Constat to make the second distribution different from the first
    x2_mu_diff = 0.35

    # Creating first distribution
    d1 = pd.DataFrame(
        {'x1': np.random.normal(mu_x1, sigma_x1, 1000), 'x2': np.random.normal(mu_x1, sigma_x1, 1000), 'type': 0})
    # FIND OUT WHAT NP.RANDOM.NORMAL DOES
    # FIND OUT WHAT pd.DataFrame does??

    # Creating the Second Distribution
    d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + x2_mu_diff,
                       'x2': np.random.normal(mu_x1, sigma_x1, 1000) + x2_mu_diff, 'type': 1})

    data = pd.concat([d1, d2], ignore_index=True)
    # FIND OUT WHAT pd.concat does??

    # From here, we can observe that the 2 distributions are linearly separable, so it's an appropriate task for our model
    # ax = sns.scatterplot(x="x1",y="x2", hue="type", data=data)
    # plt.show()

    # Create a training set to train the network

    # Splitting the data set in the training and test set
    msk = np.random.rand(len(data)) < 0.8  # WHAT DOES THIS FUNCTION DO??

    # Roughly 80% of the data will go in the training set
    train_x, train_y = data[['x1', 'x2']][msk], data.type[msk]  # HOW DOES THIS COMMAND WORK
    # Everything will go into the validation set
    test_x, test_y = data[['x1', 'x2']][~msk], data.type[~msk]  # HOW DOES THIS COMMAND WORK

    """
    #Feed Forward Neural Network

    One of the main drawbacks of the perceptron algorithm constructed above is that it can only capture linear relationships. 
    We want to be able to add any non-linearity so that we will be able to separate the space in a more complex way. That is 
    the goal with Multilayer Neural Networks. The particular case we want to study is the Feed Forward Neural Network (FFNN),
    which is a network that has only one direction, from input to output. We will also study one of the most common ways to 
    train a Feedforward NN, using a technique called backpropagation. 

    Note that, we can also introduce non-linearity by changing the activation function. Examples of non-linear activation functions
    include ReLU and Sigmoid.

    """

    # Training FFNN

    # Splitting the dataset in training and test set
    msk = np.random.rand(len(data)) < 0.8

    # Roughly 80% of data will go in the training set
    train_x, train_y = data[['x1', 'x2']][msk], data[['type']][msk].values

    # Everything else goes will go into the validation set
    test_x, test_y = data[['x1', 'x2']][~msk], data[['type']][~msk].values

    my_network = FFNN()
    my_network.fit(train_x, train_y, epochs=10000, step=0.001)

    pred_y = test_x.apply(my_network.forward, axis=1)

    # Shaping the data
    test_y_ = [i[0] for i in test_y]
    pred_y_ = [i[0] for i in pred_y]

    print('MSE: ', sklearn.metrics.mean_squared_error(test_y_, pred_y_))
    print('AUE: ', sklearn.metrics.roc_auc_score(test_y_, pred_y_))

    threshold = 0.5
    pred_y_binary = [0 if i > threshold else 1 for i in pred_y_]

    cm = confusion_matrix(test_y_, pred_y_binary, labels=[0, 1])

    print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1']))
    # How to plot this?? To make sure it is the same as in the book


main()
