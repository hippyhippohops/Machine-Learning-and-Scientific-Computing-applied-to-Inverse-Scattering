import matplotlib.pyplot
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

"""
An Introduction to Machine Learning( Supervised Learning & Unsupervised Learning)

ML is a variegated field with many different types of algorithms that try to learn in slightly different ways. We can divide
them into the following different categories according to the way the algorithm performs the learning:

(i) Supervised Learning

What is supervised learning about: The goal of supervised learning algorithms is to try to find a good approximation of the
function that is mapping inputs and outputs. To do this, it is necessary to provide both the input and the output values to 
the algorithms yourself, thus leading to the name supervised learning. The algorithm then will try to find a function that 
minimises the errors between the predictions and the actual output.

(ii) Unsupervised Learning

Unsupervised learning works with unlabeled data, so we do not need the actual output, only the input. Unsupervised learning 
algorithms tries to find patterns in the data, and react based on those commonalities, diving the input into clusters. 


(iii) Semi-supervised Learning

Semi-supervised Learning is a technique in between supervised and unsupervised learning. The aim is to reduce the cost of
gathering labeled data by extending a few labels to similar unlabeled data.

(iv) Reinforcement Learning

Reinforcement Learning is the most distinct category, with respect to the one we saw so far. Reinforcement learning algorithms
try to find a policy to maximise the sum of rewards. The policy is learned by an agent who uses it to take actions in an 
environment. The environment then returns the feedback, which the agent uses to improve it's policy. The feedback is the reward
for the action taken and it can be a positive, null, or negative number.

#FOCUSING ON SUPERVISED LEARNING

For now, we will primarily focus on supervised learning. These algorithms try to find a good approximation of 
the function that is mapping inputs  and outputs. To accomplish that, it is necessary to provide both input values and 
output values to the algorithm yourself, and it will try to find a function that minimizes the errors between the 
predictions and the actual output. This learning  phase is called training. After a model is trained, it can be used to
predict the output from unseen data. This phase is commonly regarded as scoring or predicting. 

There are 2 types of supervised learning: Classification and Regression. 

(i) Classification

Classification is a task where the output variable can assume a finite amount of elements called categories. Classification 
can be further broken down into the 3 categories: Binary Classification, Multiclass Classification, and Multilabel 
Classification. 

(ii) Regression

Regression is a task where the output variable is continuous. Some of the common regression algorithms include linear 
regression and logistic regression. 

Note that there are a lot of algorithms for supervised learning. We choose the algorithm based on the task and the data we 
have at our disposal. If we do not much data and there is already some knowledge around our problem, deep learning is probably
not the best approach to start with. 

Starting simple is always a good practise; for example, for categorisation, a good starting point can be a decision tree.
A simple decision tree that is simple to overfit is a random forest. For regression problems, linear regression is still popular,
especially in domains where it is neccessary to justify the decision taken. For other problems such as recommender systems, a good
starting point can be Matrix Factorisation. Each domain has a standard algorithm that is better to start with. 
"""

"""
METRICS

The metric chosen to evaluate the algorithm is another extremely important step in the machine learning process. You can choose a 
particular metric as the loss of the algorithm aims to minimise. Once again, we can divide the metrics by the type of the problem
we have. 

For regression, in Keras, we have the following important metrics:
(i) Mean Squared Error
(ii) Mean Absolute Value Error
(iii) Mean Absolute Percentage Error
(iv) Cosine Proximity 

For classification, in Keras, we have the following important metrics:
(i) Binary Accuracy
(ii) ROC AUC
(iii) Categorical Accuracy
(iv) Sparse Categorical Accuracy
(v) Top K Categorical Accuracy
(vi) Sparse Top K Categorical Accuracy

"""


"""
#Introducing Neural Networks

Artifical Neural Networks (ANN) are a set of bio-insprired algorithms. In particular, they are loosely inspired 
by biologica, brains; exactly like animal brains, ANN consists of simple units (neurons) connected to each other. They 
receive, process, and transmit a signal to other neurons, acting like a switch. The elements of a neural network are 
quite simple on their own. The complexity and the power of these systems come from the interaction between the elements. 
Here, we will:
(i) Learn about the following
    -> The Perceptron
    -> Feedforward Neural Networks
    -> How Backprogation is used to train FeedforwardNN
(ii) Build the following from scratch:
    -> Perceptron
    -> A Simple FeedForward Neural Network 
(iii) Learn to use Keras to train NNs
"""

"""
CONCEPT OF PERCEPTRON

The concept of the perceptron is inspired by the biological neuron. The main function of the perceptron is to decide to 
block or let a signal pass. As we draw inspiration of how neurons are inspired, let's take a look at how neurons work. 
Neurons receive a set of binary input, created by electrical signals. If the total signal surpasses a certain threshold, 
the neuron fires an output. A perceptron works the same way. 

A perceptron receives multiple pieces of inputm and this input is then multiplied by a set of weights. The sum of the 
weighted signal will then pass through an activation function. In this case, the step function acts as the activation 
function. If the total signal is greater than a certain threshold, the perceptron will either let the signal pass or not.
"""

"""
IMPLEMENTING A PERCEPTRON 

We will now look at how to build a perceptron from scratch. A single-layer perceptron that only contains one input layer 
and one output layer. There is no presence of the hidden layers. (Add images from google and 
https://www.javatpoint.com/single-layer-perceptron-in-tensorflow). Single-layer perceptrons are only capable of learning
patterns that are linearly separable. Linearly separable data is data that can be separated by a line, linear function, 
or flat hyperplane. In general, two groups of data points are separable in an n-dimensional space if they can be 
separated by an n-1 dimensional hyperplane. The learning part is the process of finding the weights that minimize the error
of the output. 
"""

#Here, we create a simple class to implement our perceptron.
"""
We know that our input data has 2 pieces of input (the coordinates in our graph) and a binary output (the type of data
point), distinguished by different colors. 
"""
class Perceptron(object):
    """
    Simple Implementation of the perception Algorithms
    """

    def __init__(self, w0=1, w1=0.1, w2=0.1):
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

    #We will use the step function as our activation function for the aritifical neuron, which will be the filter that
    #decides whether the signal should pass.
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
    def predict_boundary(self,x):
        """
        Used to predict the boundaries of our classifier
        """
        return -(self.w1*x +self.w0)/self.w2

    def fit(self,X,y,epochs=1, step=0.1,verbose=True):
        """
        Train the model given the dataset
        """
        errors = []

        for epoch in range(epochs):
            error = 0
            for i in range(0,len(X.index)):
                x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
                #The update is proportional to the step size and the error
                update = step * (target-self.predict(x1,x2))
                self.w1 += update * x1
                self.w2 += update * x2
                self.w0 += update
                error += int(update != 0.0)
            errors.append(error)
            if verbose:
                print('Epochs:{} - Error: {} - Errors from all epochs:{}'.format(epoch,error,errors))


def sigmoid(s):
    #Activation function
    return 1 / (1 + np.exp(-s))

def sigmoid_prime(s):
    # Derivative of the Sigmoid
    return sigmoid(s) * (1-sigmoid(s))

class FFNN(object):

    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        #Adding 1 as it will be our bias
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size

        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0

        #The whole weight matrix, from the inputs till the hidden layer
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        #The Final set of weights from the hidden layer till the output layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self,X):
        # Forward propgation through our network
        X['bias'] = 1 # Adding 1 to the inputs to include the bias in the weight
        self.z1 = np.dot(X, self.w1) # dot product of X (inout) and first set of 3x2 weights
        self.z2 = sigmoid(self.z1) # Activation Function
        self.z3 = np.dot(self.z2, self.w2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = sigmoid(self.z3) # final activation function
        return o

    def predict(self,X):
        return forward(self, X)

    def backward(self, X, y, output, step):
        #Backward propagation of the errors
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.o_error = y - output #error in output
        self.o_delta = self.o_error * sigmoid_prime(output) * step # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.w2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2) * step # Applying derivative of sigmoid to z2 error

        self.w1 += X.T.dot(self.z2_delta) # Adjusting first of weights
        self.w2 += self.z2.T.dot(self.o_delta) # Adjusting the second set of weights

    def fit(self, X, y, epochs=10,step=0.05):
        for epoch in range(epochs):
            X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
            output = self.forward(X)
            self.backward(X,y, output,step)



def main():
    """
    Here, we create a dataset. We will do so by sampling from 2 distinct normal distributions that we created, labeling
    the data according to the distribution.
    """
    #Inintialisg random number
    np.random.seed(11) #To play with this, we can replot with random seeds

    ### Creating the dataset

    # mean and standard deviation for the x belonging to the first class
    mu_x1, sigma_x1 = 0, 0.1

    # Constat to make the second distribution different from the first
    x2_mu_diff = 0.35 #To play with this, we can replot with different 2nd distributions

    #Creating first distribution
    d1 = pd.DataFrame({'x1':np.random.normal(mu_x1,sigma_x1,1000), 'x2': np.random.normal(mu_x1, sigma_x1,1000), 'type': 0})

    #Creating the Second Distribution
    d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000)+x2_mu_diff, 'x2': np.random.normal(mu_x1, sigma_x1, 1000)+x2_mu_diff, 'type': 1})

    data = pd.concat([d1,d2],ignore_index=True)

    #From here, we can observe that the 2 distributions are linearly separable, so it's an appropriate task for our model
    #ax = sns.scatterplot(x="x1",y="x2", hue="type", data=data)
    #plt.show()

    #Create a training set to train the network

    #Splitting the data set in the training and test set
    msk = np.random.rand(len(data)) < 0.8

    #Roughly 80% of the data will go in the training set
    train_x, train_y = data[['x1','x2']][msk], data.type[msk]
    #Everything will go into the validation set
    test_x, test_y = data[['x1', 'x2']][~msk], data.type[~msk]

    my_perceptron = Perceptron(0.1,0.1)
    my_perceptron.fit(train_x, train_y, epochs=1,step=0.005)

    #print("end")

    """
    To check the algorithm's performance, we can use the confusion matrix, which shows all of the correct predictions and
    the misclassifications. As it's a binary task,  we will have three possible options for the results: Correct, False
    positive.
    """

    pred_y = test_x.apply(lambda x: my_perceptron.predict(x.x1, x.x2),axis = 1)
    cm = confusion_matrix(test_y,pred_y, labels=[0,1])
    print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1']))

    # We want to visualise the prediction results on the input space by drawing the linear decision boundary. To accomplish
    # that, we need to add the following method in our perceptron class

    #Add decision boundary line to the scatterplot
    ax = sns.scatterplot(x="x1",y="x2",hue="type", data=data[~msk])
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = my_perceptron.predict_boundary(x_vals)
    ax.plot(x_vals, y_vals, '--', c="red")
    #print("end")
    #plt.show()

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

    #Training FFNN

    # Splitting the dataset in training and test set
    msk = np.random.rand(len(data)) < 0.8

    #Roughly 80% of data will go in the training set
    train_x, train_y = data[['x1','x2']][msk], data[['type']][msk].values

    #Everything else goes will go into the validation set
    test_x, test_y = data[['x1','x2']][~msk], data[['type']][~msk].values

    my_network = FFNN()
    my_network.fit(train_x, train_y, epochs=10000, step=0.001)

    pred_y = test_x.apply(my_network.forward, axis=1)

    #Shaping the data
    test_y_ = [i[0] for i in test_y]
    pred_y_ = [i[0] for i in pred_y]

    print('MSE: ', sklearn.metrics.mean_squared_error(test_y_,pred_y_))
    print('AUE: ', sklearn.metrics.roc_auc_score(test_y_,pred_y_))

    threshold = 0.5
    pred_y_binary = [0 if i > threshold else 1 for i in pred_y_]

    cm = confusion_matrix(test_y_, pred_y_binary, labels=[0,1])

    print(pd.DataFrame(cm, index=['True 0', 'True 1'],columns=['Predicted 0', 'Predicted 1']))
    #How to plot this?? To make sure it is the same as in the book








main()
