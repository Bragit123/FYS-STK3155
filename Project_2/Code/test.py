import numpy as np
from funcs import CostLogReg, sigmoid
from scheduler import Adam
from NN import FFNN

scheduler = Adam(0.1, 0.9, 0.999)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
t = np.array([0,0,0,1], dtype=float)
t = np.c_[t]

dim = (2,5,1)
seed = 100
classification = True

network = FFNN(dim, sigmoid, sigmoid, CostLogReg, seed, classification)
network.train(X, t, scheduler)
output = network.predict(X)

print(output)