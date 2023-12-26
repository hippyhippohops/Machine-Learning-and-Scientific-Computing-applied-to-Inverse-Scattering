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

    # Now, we will use keras in order to implement a perceptron

    """Now that we have seen how to implement a perceptron from scratch in Python and have understood the concept, we 
    can use a library to avoid re-implementing all of these algorithms. We will now use keras on top of Tensorflow to 
    implement our perceptron, by introducing some simple concepts.

    The main objective of Keras is to make the model creation more Pythonic and model-centric. There are 2 ways to 
    create a model, using either the:
    -> Sequential Class
    -> Model Class. 
    The easiest way to create a Keras model is by using the Sequential API. Note that there are some limitations that 
    come with using the Sequential API: One of it includes the fact that it is not straightforward to define models that
    may have mutliple different inputs or output sources, but this fits our purpose. 
    """

    # We start by intialising the sequential class:
    my_perceptron = keras.Sequential()

    # Then, we will have to add our input layer and specify the dimensions and a few other parameters. In our case, we
    # will add a Dense layer. This just means that all of the neurons have a connection with all of the neurons from the
    # next layer. This Dense layer is fully connected, meaning that all of the neurons have one connection with the neurons
    # from the next layer. It performs the product between the input and our set of weights, which is also the kernel,
    # of course, adding the bias if specificed. Then, the result will pass through the activation function. To initialise
    # this, we need to specify the number of neurons (1), the input dimension (2, as we have 2 variables here), the activation
    # function (linear), and the initial weight value (zero). To add the layer to the model, we use the add() method, like
    # in the input/output example:
    input_layer = keras.layers.Dense(1, input_dim=2, activation="sigmoid",
                                     kernel_initializer='zero')  # FIND OUT WHAT IS Initializer??
    my_perceptron.add(input_layer)

    # Now, it is necessary to compile our model. In this phase, we will simply define the loss function and the way we want
    # to explore the gradient, our optimizer. Keras does not supply the step function that we used before, as it's not
    # differentiable and therefore will not work with back propagation. We will use the MSE instead for simplicity. Also,
    # as a gradient descent strategy, we are going to use Stochastic Gradient Descent (SGD), which is an iterative method
    # to optimize a differentiable function. When defining the SGD, we can also specify a learning rate, which we will
    # set as 0.01:
    my_perceptron.compile(loss="mse", optimizer=SGD(lr=0.01))  # Read up on Stochastic Gradient Descent

    # After this, we only need to train our network with the fit method.
    """Add in detailed notes here"""
    my_perceptron.fit(train_x.values, train_y, nb_epoch=2, batch_size=32, shuffle=True)

    # Now, we can easily compute the AUC score
    """Read up on AUC Score"""
    pred_y = my_perceptron.predict(test_x)
    print(roc_auc_score(test_y, pred_y))


main()
