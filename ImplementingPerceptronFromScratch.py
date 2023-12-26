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

class Perceptron(object):
    """
    Simple Implementation of the perception Algorithms
    """

    # Here, we initialize the weights
    def __init__(self, w0=1, w1=0.1,
                 w2=0.1):  # Read up on Objected-Oriented Programming and how does classes work, also __init__??
        """
        We need 2 weights, one for each input, plus one extra one to represent the bias term of our equation. We will
        represent the bias as a weight that always receives an input equal to 1. This will make the optimization easier.
        """
        # Weights
        self.w0 = w0  # bias
        self.w1 = w1
        self.w2 = w2

    """
    We now add the methods to calculate the prediction to our class, which refers to the part that implements the m
    mathematical formula. Of course, at the beginning, we don't know what the weights are (that's why we actually train
    the model) but we need some values to start, so we initialise them to an arbitrary value. 
    """

    # We will use the step function as our activation function for the aritifical neuron, which will be the filter that
    # decides whether the signal should pass.
    def step_function(self, z):
        if z >= 0:
            return 1
        else:
            return 0

    """
    The input will then be summed and multiplied by the weights, so we will need to implement a method that will take 2
    pieces of input and return their weighted sum. The bias term is indicated by the term self.w0, which is always muliplied 
    by the unit. 
    """

    def weighted_sum_inputs(self, x1, x2):
        return sum([1 * self.w0, x1 * self.w1, x2 * self.w2])

    """
    Now, we implement the predict function, which uses the functions we defined in the preceding code block to calculate the 
    output of the neuron. 
    """

    def predict(self, x1, x2):
        """
        Uses the step function to determine the output
        """
        z = self.weighted_sum_inputs(x1, x2)
        return self.step_function(z)

    """
    The training phase, where we calculate the weights, is a simple process that is implemented with the following fit method.
    We need to provide this method with the input, the output, and 2 more parameters; the number of epochs and the step size.

    An epoch is a single step in training our model, and it ends when all training samples are used to update the weights.

    The step size is a parameter that helps to control the effect of new updates on the current weights. The perceptron 
    convergence theorem states that a perceptron will converge if the classes are linearly separable, regardless of the 
    learning rate. 

    In the following code block, it is possible to find the code for the method that we need to add to the perceptron class
    to do the training. 

    The training process calculates the weight update by multiplying the step size (or learning rate) by the difference 
    between the real output and the prediction. The weighted error is then multiplied by each input and added to the 
    corresponding weight. It is a simple update strategy that will allow us to divide the the region in two and classify 
    our data. This learning strategy is known as the Perceptron Learning Rule.

    """

    def predict_boundary(self, x):
        """
        Used to predict the boundaries of our classifier
        """
        return -(self.w1 * x + self.w0) / self.w2

    def fit(self, X, y, epochs=1, step=0.1, verbose=True):
        """
        Train the model given the dataset
        """
        errors = []

        for epoch in range(epochs):
            error = 0
            for i in range(0, len(X.index)):
                x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
                # The update is proportional to the step size and the error
                update = step * (target - self.predict(x1, x2))
                self.w1 += update * x1
                self.w2 += update * x2
                self.w0 += update
                error += int(update != 0.0)
            errors.append(error)
            if verbose:
                print('Epochs:{} - Error: {} - Errors from all epochs:{}'.format(epoch, error, errors))


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

    my_perceptron = Perceptron(0.1, 0.1)
    my_perceptron.fit(train_x, train_y, epochs=1, step=0.005)

    # print("end")

    """
    To check the algorithm's performance, we can use the confusion matrix, which shows all of the correct predictions and
    the misclassifications. As it's a binary task,  we will have three possible options for the results: Correct, False
    positive.
    """

    pred_y = test_x.apply(lambda x: my_perceptron.predict(x.x1, x.x2), axis=1)
    cm = confusion_matrix(test_y, pred_y, labels=[0, 1])
    print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1']))

    # We want to visualise the prediction results on the input space by drawing the linear decision boundary. To accomplish
    # that, we need to add the following method in our perceptron class

    # Add decision boundary line to the scatterplot
    ax = sns.scatterplot(x="x1", y="x2", hue="type", data=data[~msk])
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = my_perceptron.predict_boundary(x_vals)
    ax.plot(x_vals, y_vals, '--', c="red")
    # print("end")
    # plt.show()


main()
