# Code
In this folder we have all the code that needs to be run in order to produce the
results of our project.

## digits_NN.py
This code analyzes the MNIST dataset using a regular neural network built from
scratch. The code produces four plots, one for each of the following hidden
layer activation functions: identity, sigmoid, RELU and LRELU. Each plot are
heatmap plots with the validation accuracy for different learning rates and
regularization parameters. To run the code, use

> python3 digits_NN.py

## cnn_handwritten_numbers.py
This code analyzes the MNIST dataset using tensorflows convolutional neural
network. The code produces eight plots, four showing training accuracies, and
four showing test accuracies. The four plots in each category correspond to the
four different hidden layer activation functions used: identity, sigmoid, RELU
and LRELU, and each plot are heatmap plots with accuracies for different
learning rates and regularization parameters. To run the code, use

> python3 cnn_handwritten_numbers.py

## Boosting.py
This code analyzes the MNIST dataset using scikit-learns boosting methods. The
code produces three plots, one for each of the following boosting methods:
adaboost, gradient boost and XG boost. The XG boost plot is a heatmap plots,
with validation accuracies for different learning rates and regularization
parameters, while adaboost and gradient boost are barplots for different
learning rates only. To run the code, use

> python3 Boosting.py

## Background code
The rest of the code in this folder are codes that are used in the three above
mentioned programs, but are not run explicitly. These are:
- **funcs.py**, which contains different functions that are used, such as activation functions and cost functions.
- **scheduler.py**, which contains different gradient descent algorithms.
- **NN.py**, which contains the code for the regular neural network. This code
  is largely based on the lecture notes by Morten Hjort-Jensen at https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#the-neural-network
- **plotting.py**, which contains code for different plotting methods, so that
  we would not have to repeat ourselves more than necessary. This is where we
  keep the heatmap and barplot functions, that are used to create the figures in
  our report.
  