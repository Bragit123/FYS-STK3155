import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from functions import *
from ols import OLSfit
from ridge import ridgefit
from lasso import Lassofit


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

# Generate the data.
nsamples = 100
x = np.sort(np.random.rand(nsamples))
y = np.sort(np.random.rand(nsamples))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

## Cross-validation on Ridge regression


def crossvalidation(k,x,y,z,lambda_val,deg):
    kfold_ind = np.linspace(0,len(x), k+1)  #postition indices for us to divide
                                               #the training data and test data in different places
    X = FeatureMatrix(x,y,z,int(deg))

    MSE_test_array_OLS = np.zeros(k+1)
    MSE_test_array_ridge = np.zeros(k+1)
    MSE_test_array_Lasso = np.zeros(k+1)
    for i in range(k+1):
        index_array = np.array(range(int(kfold_ind[i]),int(kfold_ind[i+1])+1))
        print(type(X))
        X_test = X[int(kfold_ind[i]):int(kfold_ind[i+1])]
        X_train = X[:int(kfold_ind[i])].apppend(X[int(kfold_ind[i+1]):])
        z_test = z[int(kfold_ind[i]):int(kfold_ind[i+1])]
        z_train = z[:int(kfold_ind[i])].append(z[int(kfold_ind[i+1]):])

        X_test, X_train, z_test, z_train = Scale(X_test, X_train, z_test, z_train)

        MSE_train, MSE_test_array_OLS[i], R2_train, R2_test, beta = OLSfit(X_test, X_train, z_test, z_train,deg)
        MSE_train, MSE_test_array_ridge[i], R2_train, R2_test, beta = ridgefit(X_test, X_train, z_test, z_train,deg,lambda_val)
        MSE_train, MSE_test_array_Lasso[i], R2_train, R2_test, beta = Lassofit(X_test, X_train, z_test, z_train,deg,lambda_val)

    return np.mean(MSE_test_array_OLS), np.mean(MSE_test_array_ridge), np.mean(MSE_test_array_Lasso)

k=5 #folds
lambdas = np.array([0.0001,0.001,0.01,0.1,1]) #lambdas = np.logspace(-3, 5, nlambdas)
degs = np.linspace(1,5,5)
MSEs_test_OLS = np.zeros(len(lambdas))
MSEs_test_ridge = np.zeros(len(lambdas))
MSEs_test_Lasso = np.zeros(len(lambdas))
for i in range(len(lambdas)):
    MSE, MSEs_test_ridge[i], MSE_tests_Lasso[i] = crossvalidation(k,x,y,z,5,lambdas[i]) #MSE is throwaway
for i in range(len(degs)):
    MSEs_test_OLS[i], MSE, MSE2 = crossvalidation(k,x,y,z,int(degs[i]),0)
plt.figure()
plt.plot(degs,MSEs_test_OLS, label="MSE test")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE test")
plt.title("MSE crossvalidation, OLS")
plt.legend()
plt.savefig("crossvalridge.pdf")

plt.figure()
plt.plot(np.log(lambdas),MSEs_test_ridge, label="MSE test")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE test")
plt.title("MSE crossvalidation, Ridge")
plt.legend()
plt.savefig("crossvalridge.pdf")

plt.figure()
plt.plot(np.log(lambdas),MSEs_test_Lasso, label="MSE test")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE test")
plt.title("MSE crossvalidation, Lasso")
plt.legend()
plt.savefig("crossvalLasso.pdf")
