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

## Cross-validation
def crossvalidation(k,x,y,z,lambda_val,deg):
    kfold_ind = np.linspace(0,len(x), k+1)  #position indices for us to divide
                                               #the training data and test data in different places

    X = FeatureMatrix(x,y,z,int(deg))

    MSE_test_array_OLS = np.zeros(k)
    MSE_test_array_ridge = np.zeros(k)
    MSE_test_array_Lasso = np.zeros(k)

    for i in range(k):
        #Dividing into train-test
        X_test = X[int(kfold_ind[i]):int(kfold_ind[i+1]),:]
        X_copy = X.copy()
        X_train = np.delete(X_copy, np.array(range(int(kfold_ind[i]),int(kfold_ind[i+1]))),0)

        z_test = z[int(kfold_ind[i]):int(kfold_ind[i+1])]
        z_copy = z.copy()
        z_train = np.delete(z_copy, np.array(range(int(kfold_ind[i]),int(kfold_ind[i+1]))))

        X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test) #Scaling

        MSE_train, MSE_test_array_OLS[i], R2_train, R2_test, beta = OLSfit(X_train, X_test, z_train, z_test)  #We only want to save MSE-test
        MSE_train, MSE_test_array_ridge[i], R2_train, R2_test, beta = ridgefit(X_train, X_test, z_train, z_test,lambda_val)
        MSE_train, MSE_test_array_Lasso[i], R2_train, R2_test = Lassofit(X_train, X_test, z_train, z_test,lambda_val)

    return np.mean(MSE_test_array_OLS), np.mean(MSE_test_array_ridge), np.mean(MSE_test_array_Lasso)

k=5 #folds
lambdas = np.array([0.0001,0.001,0.01,0.1,1])
degs = np.linspace(1,5,5)
MSEs_test_OLS = np.zeros(len(degs))
MSEs_test_ridge = np.zeros(len(lambdas))
MSEs_test_Lasso = np.zeros(len(lambdas))

for i in range(len(lambdas)):
    MSE, MSEs_test_ridge[i], MSEs_test_Lasso[i] = crossvalidation(k,x,y,z,lambdas[i],5) #MSE is throwaway, deg=5
for i in range(len(degs)):
    MSEs_test_OLS[i], MSE, MSE2 = crossvalidation(k,x,y,z,0.1,int(degs[i])) #random lambda=0.1

plt.figure()
plt.plot(degs,MSEs_test_OLS, label="MSE test")
plt.xlabel("degree")
plt.ylabel("MSE test")
plt.title("MSE crossvalidation, OLS")
plt.legend()
plt.savefig("crossvalOLS.pdf")

plt.figure()
plt.plot(np.log10(lambdas),MSEs_test_ridge, label="MSE test")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE test")
plt.title("MSE crossvalidation, Ridge")
plt.legend()
plt.savefig("crossvalridge.pdf")

plt.figure()
plt.plot(np.log10(lambdas),MSEs_test_Lasso, label="MSE test")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE test")
plt.title("MSE crossvalidation, Lasso")
plt.legend()
plt.savefig("crossvalLasso.pdf")
