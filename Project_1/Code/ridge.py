import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn import linear_model
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from functions import *
plt.style.use("seaborn-v0_8")

from random import random, seed
seed = np.random.seed(200)

## Creating data set
N = 100 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

## Initiate arrays for the values that we want to compute
deg = 5 # Polynomial degree
lambda_exp_start = -20
lambda_exp_stop = -3
lambda_num = 1000

lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)
MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num
X = FeatureMatrix(x, y, z, deg) # Compute feature matrix
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) # Split into training and test data
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test) # Scale data

for i in range(lambda_num):
    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i], beta_list[i] = Ridgefit(X_train, X_test, z_train, z_test, lambdas[i])

plt.figure()
plt.title(f"Mean square error for Ridge regression with polynomial degree {deg}.")
plt.plot(np.log10(lambdas),MSE_train_array,label="MSE_train, Ridge")
plt.plot(np.log10(lambdas),MSE_test_array,label="MSE_test, Ridge")
plt.xlabel("log10lambda")
plt.ylabel("MSE")
plt.legend()
plt.savefig("MSERidge.pdf")

plt.figure()
plt.title(f"R2-values for Ridge regression with polynomial degree {deg}.")
plt.plot(np.log10(lambdas),R2_train_array,label="R2_train, Ridge")
plt.plot(np.log10(lambdas),R2_test_array,label="R2_test, Ridge")
plt.xlabel("log10lambda")
plt.ylabel("R2-score")
plt.legend()
plt.savefig("R2Ridge.pdf")