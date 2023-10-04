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
z = z + 0.1*np.random.normal(0, 1, z.shape)

k = 5


##Lasso
deg = 5
lambda_exp_start = -10
lambda_exp_stop = -3
lambda_num = 20
lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)

mse_lasso_cv = np.zeros(lambda_num)
mse_lasso_cv_std = np.zeros(lambda_num)
mse_lasso = np.zeros(lambda_num) # For Lasso without crossvalidation

X = FeatureMatrix(x, y, deg)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)
for i in range(lambda_num):
    # Lasso with crossvalidation
    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="lasso", lambda_val=lambdas[i])
    mse_lasso_cv[i] = mse_test_mean
    mse_lasso_cv_std[i] = mse_test_std

    # Lasso without crossvalidation

    mse_train, mse_test, r2_train, r2_test, beta = Lassofit(X_train, X_test, z_train, z_test, lambdas[i])
    mse_lasso[i] = mse_test


# # Plot Lasso
plt.figure()
plt.title("Mean square error of lasso regression with and without crossvalidation")
plt.xlabel("Lambda")
plt.ylabel("Mean Square Error (MSE)")
plt.plot(np.log10(lambdas), mse_lasso, label="Without crossvalidation")
plt.errorbar(np.log10(lambdas), mse_lasso_cv, mse_lasso_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend()
plt.savefig(f"cv_lasso.pdf")
