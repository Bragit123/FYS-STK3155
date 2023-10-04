import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use("seaborn-v0_8")

from functions import *

np.random.seed(200)

## Creating data set
N = 100 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z = z + 0.1*np.random.normal(0, 1, z.shape)

num_samples = 10

## Initiate arrays for the values that we want to compute
deg_num = 8
degs = np.linspace(1, deg_num, deg_num, dtype=int)

mse_train_mean = np.zeros(deg_num)
mse_train_std = np.zeros(deg_num)
mse_test_mean = np.zeros(deg_num)
mse_test_std = np.zeros(deg_num)

for i in range(deg_num):
    X = FeatureMatrix(x, y, degs[i])
    mse_train_mean[i], mse_train_std[i], mse_test_mean[i], mse_test_std[i] = Bootstrap_OLS(X, z, num_samples)

plt.figure()
plt.errorbar(degs, mse_train_mean, mse_train_std, label="MSE for training data", capsize=5, markeredgewidth=1)
plt.errorbar(degs, mse_test_mean, mse_test_std, color="r", label="MSE for test data", capsize=5, markeredgewidth=1)
plt.xlabel("Polynomial degree.")
plt.ylabel("Mean square error")
plt.legend()
plt.savefig("bootstrap_errorbar.pdf")

plt.figure()
plt.plot(degs, mse_train_mean, label="MSE for training data")
plt.plot(degs, mse_test_mean, label="MSE for test data")
plt.xlabel("Polynomial degree.")
plt.ylabel("Mean square error")
plt.legend()
plt.savefig("bootstrap.pdf")
