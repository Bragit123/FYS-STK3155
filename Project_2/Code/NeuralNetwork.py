
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
    def __init__(self, n_weights: int, activation_function: Callable[[np.ndarray], float]):
        """ Constructor function. Creates a node with n_weights random weights,
        and one random bias (both weights and bias takes values from 0 to 1).

        ## Parameters
            - n_weights (int): Number of weights to assign the node. In the
              neural network, this should be the same as the number of nodes in
              the previous layer.
            - activation_function (Callable): The activation function. This should
            be a function that takes in an array (the weighted input from the
            previous layer) and returns a float (the activation of the node).
        """
        self.weights = np.random.uniform(size=n_weights)
        self.bias = np.random.uniform()
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
    - size : int
        Number of nodes in the layer.
    - activations : ndarray
        Stores the activation of each node as an array.

    ## Methods
    -------
    - compute(x):
        Computes the activation of the node from input x (the output from the
        nodes of the previous layer).
    """
    def __init__(self, n_nodes: int, n_weights: int, activation_function: Callable[[np.ndarray], float]):
        """ Constructor function

        ## Parameters
            - n_nodes (int): Number of nodes to create in this layer.
            - n_weights (int): Number of weights these nodes should have. This
              number will be the same for all nodes in this layer, and should
              correspond to the number of nodes in the previous layer of the
              neural network.
            - activation_function (Callable): The activation function. This should
            be a function that takes in an array (the weighted input from the
            previous layer) and returns a float (the activation of the node).
            The activation_function will here be the same for all nodes in the layer.
        """

        # Initiate nodes-list and activation-array.
        self.nodes = []
        for i in range(n_nodes):
            node = Node(n_weights, activation_function)
            self.nodes.append(node)

        self.n_nodes = n_nodes
        self.n_weights = n_weights
        # Should consider wether this should really be an attribute. If yes:
        # must make sure that this is updated any time the activation of one of
        # the nodes are updated, and vice versa.
        self.activations = np.zeros(n_nodes)

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
        a = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            a[i] = self.nodes[i].compute(x)

        self.activations = a
        return self.activations

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
    def __init__(self, structure: np.ndarray, n_input: int, activation_function):
        """ Constructor function

        ## Parameters
            - layers (list): Layers to include in the neural network.
        """
        # Initiate nodes-list and activation-array.
        # Example: structure = [5, 4, 5]

        self.n_layers = np.shape(structure)[0]
        self.n_nodes = np.sum(structure)

        layer = []

        for i in range(self.n_layers):
            n_nodes = structure[i]

            if i == 0:
                n_weights = n_input
            else:
                prev_layer = layer[i-1]
                n_weights = prev_layer.n_nodes

            layer = Layer(n_nodes, n_weights, activation_function)
            self.layers.append(layer)

        self.n_input = n_input
