
import numpy as np
from typing import Callable

class Node:
    """
    Class to represent one node of our neural network.

    ## Attributes
    ----------
    - weights : ndarray
        Weights to use when computing the activation function.
    - bias : float
        Bias to use when computing the activation function.
    - activation_function : Callable
        The activation function. This should be a function that takes in an
        array (the weighted input from the previous layer) and returns a float
        (the activation of the node)
    - activation : float
        The activation of this node.

    ## Methods
    -------
    - compute(x):
        Computes the activation of the node from input x (the output from the
        nodes of the previous layer).
    """
    def __init__(self, weights: np.ndarray, bias: float, activation_function: Callable[[np.ndarray], float]):
        """ Constructor function
        
        ## Parameters
            - weights (ndarray): Weights to use when computing the activation function.
            - bias (float): Bias to use when computing the activation function.
            - activation_function (Callable): The activation function. This should
            be a function that takes in an array (the weighted input from the
            previous layer) and returns a float (the activation of the node).
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.activation = 0 # Initializes the activation of this node.
    
    def compute(self, x: np.ndarray) -> float:
        """ Computes the activation of this node, based on the input
        values (the output from the previous layer).

        ## Parameters
            - x (ndarray): Input to compute the activation (this is
              also the output of the previous nodes).
        
        ## Returns
            - activation (float): The activation of this node. This value is also stored as
            a variable in the node.
        """
        z = np.dot(self.weights, x) + self.bias
        self.activation = self.activation_function(z)
        return self.activation

class Layer:
    """
    Class to represent one layer of nodes in our neural network.

    ## Attributes
    ----------
    - nodes : list
        Nodes belonging to this layer.
    - activation : ndarray
        Stores the activation of each node as an array.
    - size : int
        Number of nodes in the layer.

    ## Methods
    -------
    - compute(x):
        Computes the activation of the node from input x (the output from the
        nodes of the previous layer).
    """
    def __init__(self, nodes: list):
        """ Constructor function
        
        ## Parameters
            - nodes (list): Nodes to include in the layer.
        """
        # Initiate nodes-list and activation-array.
        self.nodes = nodes
        self.activation = []

        # Add node and activation from each input-node
        for node in nodes:
            self.activation.append(node.activation)
        
        self.activation = np.asarray(self.activation) # Convert activation to numpy array
        
        self.size = len(nodes)
    
    def compute(self, x: np.ndarray) -> np.ndarray:
        """ Computes the activations of this layer, based on the input
        values (the output from the previous layer).

        ## Parameters
            - x (ndarray): Input to compute the activation (this is
              also the output of the previous layer).
        
        ## Returns
            - activation (ndarray): The activation of this layer. This value is also stored as
            a variable in the layer.
        """
        a = np.zeros(self.size)
        for i in range(self.size):
            a[i] = self.nodes[i].compute(x)
        
        self.activation = a
        return self.activation

class NeuralNetwork:
    """
    Class to represent a neural network.

    ## Attributes
    ----------
    - layers : list
        Layers belonging to this neural network.
    - size : int
        Number of layers in the neural network.

    ## Methods
    -------
    - feed_forward(x):
        Computes the activation of the node from input x (the output from the
        nodes of the previous layer).
    - back_propagate():
    """
    def __init__(self, layers: list, input_size: int):
        """ Constructor function
        
        ## Parameters
            - layers (list): Layers to include in the neural network.
        """
        # Initiate nodes-list and activation-array.
        self.layers = layers
        self. input_size = input_size
        self.size = len(layers)