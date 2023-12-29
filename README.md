# Neural-Networks-Deep-Learning
This repository is a collection of all my rough notes and coding done over the course of my PhD. It will be a melting pot of different concepts I have learnt, ideas that I have played with and fun things I do on the side. In general, it will be my expository/reference on Deep Learning/Neural Networks and Machine Learning cencepts in general. The references are cited at the bottom. 

# An Introduction to Machine Learning( Supervised Learning & Unsupervised Learning)

ML is a variegated field with many different types of algorithms that try to learn in slightly different ways. We can divide
them into the following different categories according to the way the algorithm performs the learning:

(i) Supervised Learning

What is supervised learning about: The goal of supervised learning algorithms is to try to find a good approximation of the
function that is mapping inputs and outputs. To do this, it is necessary to provide both the input and the output values to 
the algorithms yourself, Thus, leading to the name supervised learning. Given both the inputs and the outputs, the algorithm 
finds the estimation to the actual function that minimises the errors between the predictions and the actual output.This stage 
is called the learning stage. After the model is trained, it can be used to predict the output from unseen data. This phase is
called the predicting/scoring stage.  Examples of Supervised Learning includes: Linear Regression. 

**Notes on  Supervising Learning**

For now, we will primarily focus on supervised learning. These algorithms try to find a good approximation of 
the function that is mapping inputs  and outputs. To accomplish that, it is necessary to provide both input values and 
output values to the algorithm yourself, and it will try to find a function that minimizes the errors between the 
predictions and the actual output. This learning  phase is called training. After a model is trained, it can be used to
predict the output from unseen data. This phase is commonly regarded as scoring or predicting. 

There are 2 types of supervised learning: Classification and Regression. 

* Classification

Classification is a task where the output variable can assume a finite amount of elements called categories. Classification 
can be further broken down into the 3 categories: 
-> Binary Classification: The task of predicting whether an instance belongs either to one class or the other
-> Multiclass Classification: The task (also known as multinomial) of predicting the most probable label (class) for each 
single instance
-> Multilabel Classification: When multiple labels can be assigned to each input

* Regression

Regression is a task where the output variable is continuous. Some of the common regression algorithms include:
-> Linear regression: This finds the linear relationship between inputs and outputs 
-> Logistic regression: This finds the probability of a binary output

(ii) Unsupervised Learning

Unsupervised learning works with unlabeled data, so we do not need the actual output, only the input. Unsupervised learning 
algorithms tries to find patterns in the data, and react based on those commonalities, diving the input into clusters. Usually,
unsupervised learning is often used in conjunction with supervised learning to reduce the input space and focus the signal in the
data on a smaller number of variables. Common unsupervised learning techniques include: clustering and principal component analysis
(PCA), Independent Component Analysis (ICA) and some neural networks such as Generative Adversial Networks (GANs) and Autoencoders
(AEs). 


(iii) Semi-supervised Learning

Semi-supervised Learning is a technique in between supervised and unsupervised learning. The aim is to reduce the cost of
gathering labeled data by extending a few labels to similar unlabeled data. We can also view semi-supervised learning as a 
generalisation of supervised learning. It's aim is to reduce the cost of gathering labelled data by extending a few labels to similar
unlabelled data. Some generative models are classified as semi-supervised approaches. Semi-supervised Learning can be 
divided into:
-> Transductive Learning: This is when we want to infer the labels for unlabeled data 
-> Inductive Learning: The goal of this learning method is to infer the correct mapping from inputs to outputs 

(iv) Reinforcement Learning

Reinforcement Learning is the most distinct category, with respect to the ones we saw so far. Reinforcement learning algorithms
try to find a policy to maximise the sum of rewards. The policy is learned by an agent who uses it to take actions in an 
environment. The environment then returns the feedback, which the agent uses to improve it's policy. The feedback is the reward
for the action taken and it can be a positive, null, or negative number.

# An Introduction to Neural Networks

Artifical Neural Networks (ANN) are a set of bio-insprired algorithms. In particular, they are loosely inspired 
by biological brains; exactly like animal brains, ANN consists of simple units (neurons) connected to each other. They 
receive, process, and transmit a signal to other neurons, acting like a switch. In biology, these units are called 
neurons.

The elements of a neural network are  quite simple on their own. The complexity and the power of these systems come from 
the interaction between the elements. Here, we will:
(i) Learn about the following
    -> The Perceptron
    -> Feedforward Neural Networks
    -> How Backprogation is used to train FeedforwardNN
(ii) Build the following from scratch:
    -> Perceptron
    -> A Simple FeedForward Neural Network 
(iii) Learn to use Keras to train NNs

**Concept of a Perceptron**

As mentioned earlier, the concept of the perceptron is inspired by the biological neuron and it's main function is to 
decide to block or let a signal pass. As we draw inspiration of how neurons are inspired, let's take a look at how neurons
work. Neurons receive a set of binary input, created by electrical signals. If the total signal surpasses a certain 
threshold, the neuron fires an output. A perceptron works the same way. 

A perceptron receives multiple pieces of input and this input is then multiplied by a set of weights. The sum of the 
weighted signal will then pass through an activation function. In this case, the step function acts as the activation 
function. If the total signal is greater than a certain threshold, the perceptron will either let the signal pass or not.

%% Refer to NeuralNetworks.py to see my notes and code on how to implement perceptrons, FFNN from scratch and using Keras. 

# Finite Difference Method applied to the Helmholtz Equation

This started off as a class project (MATH715: Introduction to Applied Math I) for me. My group was tasked to do a project involving a real-life example of solving large systems of linear equations directly. Inspired by this set of notes (https://www.ljll.math.upmc.fr/frey/cours/UdC/ma691/ma691_ch6.pdf), we set off to create the following programs:
1) Lower-Upper Triangulation
2) Guassian Elimination
3) Cholesky Factorisation
4) project4 (which is the main file which discretises the ODE and calls (1)-(3) to numerically solve it)
to numerically approximate the solution to the first order differential equation: -u''(x) + e^(-x*2)u(x) = x^2 in [0,1], given the boundary conditions u(0)=u(1)=0 via the Finite Difference Method (FDM). Then, when delving further into my research in Inverse Scattering Problems, I notice that FDM in particular in useful in discretizing the Helmholtz Equation (SWITCHNET: A NEURAL NETWORK MODEL FOR FORWARD AND INVERSE SCATTERING PROBLEMS). So, here I would like to expand this work into trying to numerically approximate the solutions to the Helmholtz Equation given different Sommerfeld Radiaion Condition/Rayleigh Conditions.

To-do list:
1) Change the code in project4 to numerically solve the Helmholtz Equation for different radiating conditions
2) Explore using Finite Elements Method, Finite Volume Method instead of FDM. What are the advantages/disadvantages of each of this?
3) Explore speed of getting the solution, accuracy and do some stability analysis. 
