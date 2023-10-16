
import numpy as np
from GradientDescentSolver import Gradient_Descent
from NeuralNetwork import Node, Layer

np.random.seed(100)

def f(x):
    return 10*x

n_nodes = 5
nodes = []

n = 5

for i in range(n_nodes):
    w = np.random.rand(n)
    b = 2
    node = Node(w, b, f)
    nodes.append(node)

layer = Layer(nodes)

x = np.random.rand(n)
z_values = np.zeros(n_nodes)

for i in range(layer.size):
    w = nodes[i].weights
    b = nodes[i].bias
    z_values[i] = np.dot(w, x) + b

y = layer.compute(x)

print("z_values")
print(z_values)
print("layer.activation")
print(layer.activation)