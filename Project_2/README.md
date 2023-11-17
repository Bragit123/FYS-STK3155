# Project 2
This is a repository for the programs we run in project 2. "Code" consists of all the python programs, and "Figures" consist of all the plots we produce. Note that not all the plots contained in this folder is used in the report. 

The program "scheduler.py" contains the various gradient descent and stochastic gradient descent tuning algorithms with momentum. The tuning algorithms implemented are Adagrad, RMS-prop and Adam. Much of this code is borrowed from the code we got from lectures https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html.

The program "NN.py" contains the class "FFNN", which is the neural network. This code has a "feedforward" and "predict" function which is used to calculate the output the network predicts from a given input. The program also has a "backpropagate" and a "train" function which are used to train the weights and biases of the neural network. It also has a function that resets the weights "reset_weights". This code is also mostly borrowed from https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html, with some changes, for example in the way we calculate gradients, using jax, and we also got rid of some functions we didn't need.

"funcs.py" consists various activation and cost functions that are used for our neural network.

"plotting.py" contains functions for plotting regular plots, heatmap plots and barplots, which we use in our report. "funcs.py" contains various cost and activation functions.

In "SGD" we run the gradient descent code from "scheduler", using all the different learning algorithms, and plotting R$^2$-score and MSE (which we don't use). "RegressionFranke" uses the neural network code to fit the twodimensional Franke Function. "FFNN_breast_center"  uses the neural network code on the Wisconsin breast cancer data set (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) to train and then predict based on the input data whether a tumor is benign or malignant. "logisticRegression.py" consists the code for logistic regression on the breast cancer data.

To run any program you simply make your way to the right directory after cloning the repository, and write "python3 "program name"" in your terminal.
