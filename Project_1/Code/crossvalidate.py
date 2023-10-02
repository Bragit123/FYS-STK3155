import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use("seaborn-v0_8")

from functions import *

np.random.seed(200)

# Generate the data
nsamples = 100
x = np.sort(np.random.rand(nsamples))
y = np.sort(np.random.rand(nsamples))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

# Initiate some values
deg_min = 2
deg_max = 8
deg_num = deg_max-deg_min+1
degs = np.linspace(deg_min, deg_max, deg_num, dtype=int)
k = 5
lambda_val = 1e-12

mse_ols_cv = np.zeros(deg_num)
mse_ols_cv_std = np.zeros(deg_num)

mse_ols = np.zeros(deg_num)

for i in range(deg_num):
    X = FeatureMatrix(x, y, z, degs[i])
    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="ols")
    mse_ols_cv[i] = mse_test_mean
    mse_ols_cv_std[i] = mse_test_std

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)
    mse_train, mse_test, r2_train, r2_test, beta = OLSfit(X_train, X_test, z_train, z_test)
    mse_ols[i] = mse_test

plt.figure()
plt.xlabel("Number of folds k for cross validation")
plt.ylabel("Mean Square Error (MSE)")
plt.plot(degs, mse_ols, label="OLS")
plt.errorbar(degs, mse_ols_cv, mse_ols_cv_std, label="OLS with crossvalidation", capsize=5, markeredgewidth=1)
plt.legend()
plt.savefig(f"cv_ols.pdf")