import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use("seaborn-v0_8")

from functions import *

np.random.seed(200)

# Generate the data
nsamples = 30
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

## OLS
mse_ols_cv = np.zeros(deg_num)
mse_ols_cv_std = np.zeros(deg_num)
mse_ols = np.zeros(deg_num) # For OLS without crossvalidation
for i in range(deg_num):
    X = FeatureMatrix(x, y, degs[i])

    # OLS with crossvalidation
    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="ols")
    mse_ols_cv[i] = mse_test_mean
    mse_ols_cv_std[i] = mse_test_std

    # OLS without crossvalidation
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)
    mse_train, mse_test, r2_train, r2_test, beta = OLSfit(X_train, X_test, z_train, z_test)
    mse_ols[i] = mse_test

## Ridge and Lasso
deg = 5
lambda_exp_start = -10
lambda_exp_stop = -3
lambda_num = 30
lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)

mse_ridge_cv = np.zeros(lambda_num)
mse_ridge_cv_std = np.zeros(lambda_num)
mse_ridge = np.zeros(lambda_num) # For Ridge without crossvalidation

mse_lasso_cv = np.zeros(lambda_num)
mse_lasso_cv_std = np.zeros(lambda_num)
mse_lasso = np.zeros(lambda_num) # For Lasso without crossvalidation

X = FeatureMatrix(x, y, deg)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)
for i in range(lambda_num):
    # Ridge and Lasso with crossvalidation
    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="ridge", lambda_val=lambdas[i])
    mse_ridge_cv[i] = mse_test_mean
    mse_ridge_cv_std[i] = mse_test_std

    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="lasso", lambda_val=lambdas[i])
    mse_lasso_cv[i] = mse_test_mean
    mse_lasso_cv_std[i] = mse_test_std

    # Ridge and Lasso without crossvalidation
    mse_train, mse_test, r2_train, r2_test, beta = Ridgefit(X_train, X_test, z_train, z_test, lambdas[i])
    mse_ridge[i] = mse_test

    mse_train, mse_test, r2_train, r2_test = Lassofit(X_train, X_test, z_train, z_test, lambdas[i])
    mse_lasso[i] = mse_test

# Plot OLS
plt.figure()
plt.title("Mean square error of OLS regression with and without crossvalidation")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean Square Error (MSE)")
plt.plot(degs, mse_ols, label="Without crossvalidation")
plt.errorbar(degs, mse_ols_cv, mse_ols_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend()
plt.savefig(f"cv_ols.pdf")

# Plot Ridge
plt.figure()
plt.title("Mean square error of ridge regression with and without crossvalidation")
plt.xlabel("Lambda")
plt.ylabel("Mean Square Error (MSE)")
plt.plot(np.log10(lambdas), mse_ridge, label="Without crossvalidation")
plt.errorbar(np.log10(lambdas), mse_ridge_cv, mse_ridge_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend()
plt.savefig(f"cv_ridge.pdf")

# # Plot Lasso
plt.figure()
plt.title("Mean square error of lasso regression with and without crossvalidation")
plt.xlabel("Lambda")
plt.ylabel("Mean Square Error (MSE)")
plt.plot(np.log10(lambdas), mse_lasso, label="Without crossvalidation")
plt.errorbar(np.log10(lambdas), mse_lasso_cv, mse_lasso_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend()
plt.savefig(f"cv_lasso.pdf")
